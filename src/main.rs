use clap::Parser;
use jev_local::{app, Backend, Config};
use std::net::SocketAddr;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(version, about = "Jev System One compatible gateway: mock, local Laya, or hosted Jev")]
struct Cli {
    #[arg(long, env="JEV_LOCAL_PROVIDER", default_value="mock")]
    provider: String,
    #[arg(long, env="JEV_LOCAL_BIND", default_value="127.0.0.1:8080")]
    bind: SocketAddr,
    #[arg(long, env="LAYA_BASE_URL", default_value="http://127.0.0.1:8000")]
    laya_url: String,
    #[arg(long, env="JEV_BASE_URL", default_value="https://jev-api.com")]
    jev_url: String,
    #[arg(long, env="JEV_API_KEY")]
    jev_api_key: Option<String>,
    #[arg(long, env="JEV_LOCAL_API_KEY")]
    gateway_api_key: Option<String>,
    #[arg(long, env="BENCHMARK_DATASET_DIR")]
    benchmark_dataset_dir: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(),Box<dyn std::error::Error>> {
    tracing_subscriber::fmt().with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into())).init();
    let args = Cli::parse();
    let backend=Backend::parse(&args.provider)?;
    if backend == Backend::Jev && args.jev_api_key.is_none() { return Err("--jev-api-key or JEV_API_KEY required for hosted Jev".into()); }
    let config=Config {backend, laya_url:args.laya_url, jev_url:args.jev_url, jev_key:args.jev_api_key, gateway_key:args.gateway_api_key, benchmark_dataset_dir: args.benchmark_dataset_dir};
    let listener=tokio::net::TcpListener::bind(args.bind).await?;
    println!("jev-local listening on http://{} (provider: {})",args.bind,args.provider);
    axum::serve(listener,app(config)).await?;
    Ok(())
}
