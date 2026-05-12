use anyhow::Result;
use clap::Parser;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

#[derive(Parser)]
struct Cli {
    #[arg(short, long, default_value = "world")]
    name: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Greeting {
    message: String,
}

#[derive(Debug, Error)]
enum AppError {
    #[error("invalid name: {0}")]
    InvalidName(String),
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let re = Regex::new(r"^\w+$")?;
    if !re.is_match(&cli.name) {
        return Err(AppError::InvalidName(cli.name).into());
    }
    let greeting = Greeting { message: format!("Hello, {}!", cli.name) };
    println!("{}", json!(greeting));
    Ok(())
}
