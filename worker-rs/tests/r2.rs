//! `fvstudio::r2::R2Store`'s object operations against a tiny in-process S3
//! (no signature check there); the signer's vectors are in `src/r2.rs`.

mod common;

use common::FakeS3;
use fvstudio::r2::{ObjectStore, R2Config, R2Error, R2Store, UNSIGNED_PAYLOAD};

fn store(endpoint: &str) -> R2Store {
    R2Store::with_sleep(R2Config { endpoint: endpoint.into(), access_key_id: "AKIDEXAMPLE".into(), secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(), bucket: "studio".into() }, Box::new(|_| {}))
}

#[test]
fn round_trip_put_get_head_list() {
    let f = FakeS3::start();
    let s = store(&f.fake.base());
    let dir = fvstudio::tempdir("r2");
    let big = dir.join("video.mp4");
    let data: Vec<u8> = (0..3 * 1024 * 1024 + 123u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
    std::fs::write(&big, &data).unwrap();
    let small = dir.join("job.json");
    std::fs::write(&small, "{\"a\": 1}\n").unwrap();

    s.put_file("studio", "jobs/j1/video.mp4", &big, None).unwrap();
    s.put_file("studio", "jobs/j1/job.json", &small, Some("application/json")).unwrap();
    s.put_file("studio", "jobs/j1/ü thumb.jpg", &small, None).unwrap();
    s.put_file("studio", "uploads/x.bin", &small, None).unwrap();
    let st = f.state.lock().unwrap();
    let put: Vec<_> = st.log.iter().filter(|e| e.0 == "PUT").cloned().collect();
    assert_eq!(put[0].1, "/studio/jobs/j1/video.mp4");
    assert_eq!(put[0].2["content-type"], "video/mp4");
    assert_eq!(put[0].2["x-amz-content-sha256"], UNSIGNED_PAYLOAD);
    assert_eq!(put[0].2["content-length"], data.len().to_string());
    assert!(put[0].2["authorization"].starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"));
    assert_eq!(put[1].2["content-type"], "application/json");
    assert_eq!(put[2].1, "/studio/jobs/j1/%C3%BC%20thumb.jpg");
    assert_eq!(st.objects[&("studio".to_string(), "jobs/j1/video.mp4".to_string())].0, data);
    drop(st);

    assert_eq!(s.head_size("studio", "jobs/j1/video.mp4").unwrap(), Some(data.len() as u64));
    assert_eq!(s.head_size("studio", "jobs/j1/missing.mp4").unwrap(), None);
    assert_eq!(s.get_bytes("studio", "jobs/j1/job.json").unwrap(), b"{\"a\": 1}\n");

    let out = dir.join("dl/video.mp4");
    assert_eq!(s.get_file("studio", "jobs/j1/video.mp4", &out).unwrap(), data.len() as u64);
    assert_eq!(std::fs::read(&out).unwrap(), data);
    assert!(!dir.join("dl/video.mp4.part").exists());

    assert_eq!(s.list_keys("studio", "jobs/j1/").unwrap(), vec![("jobs/j1/job.json".to_string(), 9), ("jobs/j1/video.mp4".to_string(), data.len() as u64), ("jobs/j1/ü thumb.jpg".to_string(), 9)]);
    let st = f.state.lock().unwrap();
    let lists: Vec<_> = st.log.iter().filter(|e| e.0 == "GET" && e.1.contains("list-type=2")).collect();
    assert_eq!(lists.len(), 2);
    assert!(lists[1].1.contains("continuation-token=2") && lists[0].1.contains("prefix=jobs%2Fj1%2F"));
    drop(st);
    assert_eq!(s.list_keys("studio", "nothing/").unwrap(), vec![]);
}

#[test]
fn get_file_404_and_retry_on_503() {
    let f = FakeS3::start();
    let s = store(&f.fake.base());
    let dir = fvstudio::tempdir("r2-404");
    let e = s.get_file("studio", "jobs/none", &dir.join("none")).unwrap_err();
    assert_eq!(e.downcast_ref::<R2Error>().unwrap().status, 404);
    assert!(!dir.join("none.part").exists());
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);

    f.state.lock().unwrap().objects.insert(("studio".into(), "k".into()), (b"hello".to_vec(), Some("text/plain".into())));
    f.state.lock().unwrap().fail_next = 2;
    assert_eq!(s.get_bytes("studio", "k").unwrap(), b"hello");
    assert_eq!(f.state.lock().unwrap().log.iter().filter(|e| e.1 == "/studio/k").count(), 3);

    f.state.lock().unwrap().fail_next = 10;
    let e = s.get_bytes("studio", "k").unwrap_err();
    assert_eq!(e.downcast_ref::<R2Error>().unwrap().status, 503);
}
