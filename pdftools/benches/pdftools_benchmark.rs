use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use pdftools::parse::extract_text;
use std::{fs, hint::black_box, time::Duration};

fn criterion_benchmark(c: &mut Criterion) {
    let files = fs::read_dir(format!("{}/assets/bench", env!("CARGO_MANIFEST_DIR")))
        .expect("Failed to read benchmark directory.")
        .map(|f| f.unwrap().file_name())
        .filter(|f| f.to_str().unwrap().ends_with(".pdf"))
        .collect::<Vec<_>>();

    let sizes = files
        .iter()
        .map(|f| f.clone().into_string().unwrap())
        .map(|f| f.split('.').next().unwrap().to_string())
        .map(|f| f.parse::<u64>().unwrap())
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("papers");
    group.measurement_time(Duration::from_secs(20));

    sizes.iter().for_each(|f| {
        let file_path = format!("{}/assets/bench/{f}.pdf", env!("CARGO_MANIFEST_DIR"));

        group.throughput(criterion::Throughput::Elements(*f));
        group.bench_with_input(BenchmarkId::from_parameter(*f), f, |b, _| {
            b.iter(|| extract_text(black_box(&file_path)))
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
