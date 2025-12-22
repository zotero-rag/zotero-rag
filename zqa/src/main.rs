use dotenv::dotenv;

#[tokio::main]
async fn main() {
    dotenv().ok();

    if let Err(e) = zqa::run().await {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
