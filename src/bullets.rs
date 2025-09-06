use std::collections::HashMap;
use std::fmt::Debug;

use crate::{Error, MountID};

////////////////////////////////////////// MarkdownList ///////////////////////////////////////////

/// A markdown list structure for virtual filesystem storage.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct MarkdownList {
    pub mount_id: MountID,
    pub path: String,
    pub sections: HashMap<String, Vec<String>>, // section header -> bullet points
}

impl MarkdownList {
    /// Create a new markdown list.
    pub fn new(mount_id: MountID, path: String) -> Self {
        MarkdownList {
            mount_id,
            path,
            sections: HashMap::new(),
        }
    }

    /// Add a bullet point to a section.
    pub fn add_bullet(&mut self, section: String, bullet: String) {
        self.sections.entry(section).or_default().push(bullet);
    }

    /// Remove a bullet point from a section.
    pub fn remove_bullet(&mut self, section: &str, index: usize) -> Option<String> {
        Some(self.sections.get_mut(section)?.remove(index))
    }

    /// Create a section.
    pub fn create_section(&mut self, section: String) {
        self.sections.entry(section).or_default();
    }

    /// Delete a section.
    pub fn delete_section(&mut self, section: &str) -> Option<Vec<String>> {
        self.sections.remove(section)
    }

    /// Parse from markdown text.
    pub fn from_markdown(mount_id: MountID, path: String, markdown: &str) -> Result<Self, Error> {
        let mut list = MarkdownList::new(mount_id, path);
        let mut current_section = String::new();
        let mut current_bullet = String::new();
        let mut in_bullet = false;

        for line in markdown.lines() {
            let trimmed = line.trim();

            if trimmed.starts_with('#') {
                // Check for multiple # characters (##, ###, etc.)
                let hash_count = trimmed.chars().take_while(|&c| c == '#').count();
                if hash_count > 1 {
                    return Err(Error::InvalidSequence(format!(
                        "Only single-level headers (#) are allowed, found {} levels",
                        hash_count
                    )));
                }

                // Save current bullet if we have one (allow empty section for section-less bullets)
                if in_bullet && !current_bullet.trim().is_empty() {
                    list.add_bullet(current_section.clone(), current_bullet.trim().to_string());
                    current_bullet.clear();
                    in_bullet = false;
                }

                // New section header
                current_section = trimmed[1..].trim().to_string();
                list.create_section(current_section.clone());
            } else if trimmed.starts_with('-') || trimmed.starts_with('*') {
                // Save current bullet if we have one (allow empty section for section-less bullets)
                if in_bullet && !current_bullet.trim().is_empty() {
                    list.add_bullet(current_section.clone(), current_bullet.trim().to_string());
                }

                // Start new bullet point
                current_bullet = trimmed[1..].trim().to_string();
                in_bullet = true;
            } else if in_bullet {
                // Continue current bullet point (multi-line)
                if !current_bullet.is_empty() {
                    current_bullet.push(' ');
                }
                current_bullet.push_str(trimmed);
            }
        }

        // Save the last bullet if we have one (allow empty section for section-less bullets)
        if in_bullet && !current_bullet.trim().is_empty() {
            list.add_bullet(current_section.clone(), current_bullet.trim().to_string());
        }

        Ok(list)
    }

    /// Serialize to markdown format.
    pub fn to_markdown(&self) -> String {
        let mut output = String::new();

        for (section, bullets) in &self.sections {
            output.push_str(&format!("# {section}\n"));
            for bullet in bullets {
                output.push_str(&format!("- {bullet}\n"));
            }
            output.push('\n');
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_single_line_bullets() {
        let markdown = r#"# Section 1
- First bullet
- Second bullet
* Third bullet

# Section 2
- Another bullet"#;

        let list = MarkdownList::from_markdown(
            MountID::generate().unwrap(),
            "test.md".to_string(),
            markdown,
        )
        .expect("Should parse successfully");

        assert_eq!(list.sections.get("Section 1").unwrap().len(), 3);
        assert_eq!(list.sections.get("Section 1").unwrap()[0], "First bullet");
        assert_eq!(list.sections.get("Section 1").unwrap()[1], "Second bullet");
        assert_eq!(list.sections.get("Section 1").unwrap()[2], "Third bullet");
        assert_eq!(list.sections.get("Section 2").unwrap().len(), 1);
        assert_eq!(list.sections.get("Section 2").unwrap()[0], "Another bullet");
    }

    #[test]
    fn parse_multi_line_bullets() {
        let markdown = r#"# Tasks
- First task that
  spans multiple
  lines
- Second task
  also multi-line
* Third task is
  a single liner continued"#;

        let list = MarkdownList::from_markdown(
            MountID::generate().unwrap(),
            "test.md".to_string(),
            markdown,
        )
        .expect("Should parse successfully");

        assert_eq!(list.sections.get("Tasks").unwrap().len(), 3);
        assert_eq!(
            list.sections.get("Tasks").unwrap()[0],
            "First task that spans multiple lines"
        );
        assert_eq!(
            list.sections.get("Tasks").unwrap()[1],
            "Second task also multi-line"
        );
        assert_eq!(
            list.sections.get("Tasks").unwrap()[2],
            "Third task is a single liner continued"
        );
    }

    #[test]
    fn reject_multi_level_headers() {
        let markdown = r#"# Valid Header
- Bullet 1

## Invalid Header
- Bullet 2"#;

        let result = MarkdownList::from_markdown(
            MountID::generate().unwrap(),
            "test.md".to_string(),
            markdown,
        );
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Only single-level headers"));
        assert!(error_msg.contains("found 2 levels"));
    }

    #[test]
    fn reject_deep_headers() {
        let markdown = r#"# Valid
- Item

### Three levels"#;

        let result = MarkdownList::from_markdown(
            MountID::generate().unwrap(),
            "test.md".to_string(),
            markdown,
        );
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("found 3 levels"));
    }

    #[test]
    fn parse_section_less_bullets() {
        let markdown = r#"- First bullet without a section
- Second bullet without a section
* Third bullet

# Section Added Later
- Bullet in section"#;

        let list = MarkdownList::from_markdown(
            MountID::generate().unwrap(),
            "test.md".to_string(),
            markdown,
        )
        .expect("Should parse successfully");

        // Check that bullets without section go to empty string section
        assert_eq!(list.sections.get("").unwrap().len(), 3);
        assert_eq!(
            list.sections.get("").unwrap()[0],
            "First bullet without a section"
        );
        assert_eq!(
            list.sections.get("").unwrap()[1],
            "Second bullet without a section"
        );
        assert_eq!(list.sections.get("").unwrap()[2], "Third bullet");

        // Check section with bullets
        assert_eq!(list.sections.get("Section Added Later").unwrap().len(), 1);
        assert_eq!(
            list.sections.get("Section Added Later").unwrap()[0],
            "Bullet in section"
        );
    }

    #[test]
    fn round_trip_conversion() {
        let markdown = r#"# Section A
- Item 1
- Item 2

# Section B
- Item 3"#;

        let list = MarkdownList::from_markdown(
            MountID::generate().unwrap(),
            "test.md".to_string(),
            markdown,
        )
        .expect("Should parse successfully");

        let output = list.to_markdown();

        // Parse the output again to verify it's valid
        let list2 = MarkdownList::from_markdown(
            MountID::generate().unwrap(),
            "test.md".to_string(),
            &output,
        )
        .expect("Round-trip should work");

        assert_eq!(list.sections.len(), list2.sections.len());
        for (section, bullets) in &list.sections {
            assert_eq!(bullets, list2.sections.get(section).unwrap());
        }
    }
}
