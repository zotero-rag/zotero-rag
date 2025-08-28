use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pdftools::parse::extract_text;
use std::{fs, hint::black_box, time::Duration};

fn criterion_benchmark(c: &mut Criterion) {
    let files_and_sizes = fs::read_dir(format!("{}/assets/bench", env!("CARGO_MANIFEST_DIR")))
        .expect("Failed to read benchmark directory.")
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("pdf") {
                return None;
            }
            path.file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.parse::<u64>().ok())
                .map(|size| (path, size))
        })
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("papers");
    group.measurement_time(Duration::from_secs(20));

    files_and_sizes.iter().for_each(|(path, size)| {
        let file_path = path.to_str().unwrap();
        group.throughput(criterion::Throughput::Elements(*size));
        group.bench_with_input(BenchmarkId::from_parameter(*size), file_path, |b, p| {
            b.iter(|| extract_text(black_box(p)))
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
