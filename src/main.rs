use clap::Parser;
use jev_local::{backends::RegistryConfig, try_app, Backend, Config};
use std::net::SocketAddr;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Decision model gateway: registered model servers, simulations, and benchmarks"
)]
struct Cli {
    #[arg(long, env = "JEV_LOCAL_PROVIDER")]
    provider: Option<String>,
    #[arg(long, env = "JEV_LOCAL_BIND", default_value = "127.0.0.1:8080")]
    bind: SocketAddr,
    #[arg(long, env = "LAYA_BASE_URL", default_value = "http://127.0.0.1:8000")]
    laya_url: String,
    #[arg(long, env = "JEV_BASE_URL", default_value = "https://jev-api.com")]
    jev_url: String,
    #[arg(long, env = "JEV_API_KEY")]
    jev_api_key: Option<String>,
    #[arg(long, env = "JEV_LOCAL_API_KEY")]
    gateway_api_key: Option<String>,
    #[arg(long, env = "BENCHMARK_DATASET_DIR")]
    benchmark_dataset_dir: Option<std::path::PathBuf>,
    #[arg(long, env = "DECISION_BACKENDS_FILE")]
    backends_file: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let args = Cli::parse();
    let backend = Backend::parse(args.provider.as_deref().unwrap_or("mock"))?;
    let registry = if let Some(path) = args.backends_file {
        use tokio::io::AsyncReadExt;
        let file = tokio::fs::File::open(path).await?;
        let mut bytes = Vec::new();
        file.take(4 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .await?;
        if bytes.len() > 4 * 1024 * 1024 {
            return Err("backend config must be at most 4 MiB".into());
        }
        let mut registry: RegistryConfig = serde_json::from_slice(&bytes)?;
        if let Some(provider) = args.provider {
            registry.default_backend = provider;
        }
        Some(registry)
    } else {
        None
    };
    if registry.is_none()
        && backend == Backend::Jev
        && !args
            .jev_api_key
            .as_deref()
            .is_some_and(|v| !v.trim().is_empty())
    {
        return Err("--jev-api-key or JEV_API_KEY required for hosted Jev".into());
    }
    let config = Config {
        backend,
        laya_url: args.laya_url,
        jev_url: args.jev_url,
        jev_key: args.jev_api_key,
        gateway_key: args.gateway_api_key,
        benchmark_dataset_dir: args.benchmark_dataset_dir,
        registry,
    };
    let application = try_app(config)?;
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    println!("jev-local listening on http://{}", args.bind);
    axum::serve(listener, application).await?;
    Ok(())
}
