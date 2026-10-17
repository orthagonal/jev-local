use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
use tokio::io::AsyncReadExt;

const MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;
const BUILTINS: [(&str, &str); 2] = [
    (
        "support-triage",
        include_str!("../../datasets/support-triage.json"),
    ),
    ("sentiment", include_str!("../../datasets/sentiment.json")),
];

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub id: String,
    pub version: String,
    #[serde(default)]
    pub description: String,
    pub cases: Vec<Case>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub id: String,
    pub request: Value,
    pub expected: BTreeMap<String, Value>,
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum Selection {
    Named(String),
    Inline(Dataset),
}

pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}

impl Dataset {
    pub fn validate(&self) -> Result<(), String> {
        if !valid_id(&self.id) || self.version.trim().is_empty() {
            return Err(
                "dataset requires an id of 1..64 letters/digits/_/- and a nonempty version".into(),
            );
        }
        if self.cases.is_empty() || self.cases.len() > 1000 {
            return Err("dataset must contain 1..1000 cases".into());
        }
        let mut ids = BTreeSet::new();
        for case in &self.cases {
            if !valid_id(&case.id) || !ids.insert(&case.id) {
                return Err("case ids must be valid and unique".into());
            }
            crate::validate(&case.request).map_err(|e| format!("case {}: {e}", case.id))?;
            let questions = case.request["questions"].as_object().unwrap();
            if questions.len() > 64
                || questions.len() != case.expected.len()
                || questions.keys().any(|key| !case.expected.contains_key(key))
            {
                return Err(format!(
                    "case {}: require 1..64 questions with exactly one expected label each",
                    case.id
                ));
            }
            if let Some(model) = case.request.get("model") {
                if !model.as_str().is_some_and(|s| !s.trim().is_empty()) {
                    return Err(format!("case {}: model must be a nonempty string", case.id));
                }
            }
            for (id, question) in questions {
                if !question["instructions"]
                    .as_str()
                    .is_some_and(|s| !s.trim().is_empty())
                {
                    return Err(format!(
                        "case {} question {id}: instructions must be a nonempty string",
                        case.id
                    ));
                }
                let label = &case.expected[id];
                let valid = match question["type"].as_str().unwrap() {
                    "choice" => {
                        let options = question["criteria"].as_object().unwrap();
                        options.len() <= 255
                            && label.as_str().is_some_and(|s| options.contains_key(s))
                    }
                    "score" => {
                        let levels = question["criteria"].as_array().unwrap().len();
                        levels <= 32 && label.as_u64().is_some_and(|v| v < levels as u64)
                    }
                    "noul" => label.is_boolean(),
                    _ => unreachable!(),
                };
                if !valid {
                    return Err(format!("case {} question {id}: label must be a choice key, zero-based score index, or noul boolean; max 255 choices/32 score levels", case.id));
                }
            }
        }
        Ok(())
    }

    pub fn question_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for case in &self.cases {
            for question in case.request["questions"].as_object().unwrap().values() {
                *counts
                    .entry(question["type"].as_str().unwrap().to_owned())
                    .or_default() += 1;
            }
        }
        counts
    }

    pub fn metadata(&self) -> Value {
        serde_json::json!({"id": self.id, "version": self.version,
            "description": self.description, "cases": self.cases.len(),
            "questions_by_type": self.question_counts()})
    }
}

fn parse(text: &str) -> Result<Dataset, String> {
    let dataset: Dataset =
        serde_json::from_str(text).map_err(|_| "invalid dataset JSON or schema".to_owned())?;
    dataset.validate()?;
    Ok(dataset)
}

pub async fn load(id: &str, directory: Option<&Path>) -> Result<Dataset, String> {
    if !valid_id(id) {
        return Err("invalid dataset id".into());
    }
    if let Some((_, text)) = BUILTINS.iter().find(|(name, _)| *name == id) {
        if let Some(directory) = directory {
            if tokio::fs::try_exists(directory.join(format!("{id}.json")))
                .await
                .map_err(|_| "cannot inspect benchmark dataset directory".to_owned())?
            {
                return Err("custom datasets cannot shadow built-in dataset ids".into());
            }
        }
        return parse(text);
    }
    let directory = directory.ok_or_else(|| format!("unknown dataset: {id}"))?;
    let root = tokio::fs::canonicalize(directory)
        .await
        .map_err(|_| "cannot open benchmark dataset directory".to_owned())?;
    let path = tokio::fs::canonicalize(root.join(format!("{id}.json")))
        .await
        .map_err(|_| format!("unknown dataset: {id}"))?;
    if !path.starts_with(&root) {
        return Err("dataset file must be inside the configured directory".into());
    }
    let file = tokio::fs::File::open(path)
        .await
        .map_err(|_| format!("cannot open dataset: {id}"))?;
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .await
        .map_err(|_| format!("cannot read dataset: {id}"))?;
    if bytes.len() as u64 > MAX_FILE_BYTES {
        return Err("dataset files must be at most 4 MiB".into());
    }
    let text = std::str::from_utf8(&bytes).map_err(|_| "dataset must be UTF-8".to_owned())?;
    let dataset = parse(text)?;
    if dataset.id != id {
        return Err("dataset id must match its filename".into());
    }
    Ok(dataset)
}

pub async fn catalog(directory: Option<&Path>) -> Result<Vec<Value>, String> {
    let mut ids: BTreeSet<String> = BUILTINS.iter().map(|(id, _)| (*id).into()).collect();
    if let Some(directory) = directory {
        let mut entries = tokio::fs::read_dir(directory)
            .await
            .map_err(|_| "cannot read benchmark dataset directory".to_owned())?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|_| "cannot list benchmark datasets".to_owned())?
        {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let id = path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or("invalid dataset filename")?;
            if !valid_id(id) || !ids.insert(id.to_owned()) {
                return Err("dataset filenames must be valid, unique ids and cannot shadow built-in datasets".into());
            }
        }
    }
    let mut output = Vec::new();
    for id in ids {
        output.push(load(&id, directory).await?.metadata());
    }
    Ok(output)
}
