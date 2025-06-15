use ftail::Ftail;
use rag::vector::lance::create_initial_table;
use std::env;
use zqa::library_to_arrow;

/* This is the main integration test ensuring the whole thing works together. Right now, this is
 * marked as ignore, mostly because we do not yet have a Zotero mocker. When we do, this should
 * actually be enabled in CI even if it costs a little bit to run so that we have guarantees about
 * not breaking existing functionality. */
#[ignore]
#[tokio::test]
async fn test_integration_works() {
    Ftail::new().console(log::LevelFilter::Info).init().unwrap();

    if env::var("CI").is_ok() {
        // Skip this test in CI environments
        return;
    }

    let batch_iter = library_to_arrow(Some(0), Some(5));

    assert!(batch_iter.is_ok());
    let batch_iter = batch_iter.unwrap();
    let db = create_initial_table(batch_iter).await;

    assert!(db.is_ok());
}
