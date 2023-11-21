use clap::Parser;
use face_tracker::ErrorWrapper;
use opencv::{highgui, prelude::*, videoio};
use std::{thread, time::Duration};
use zenoh::prelude::r#async::*;

#[derive(Parser, Debug)]
#[command()]
struct Args {
    /// Endpoints to connect to.
    #[clap(short = 'e', long)]
    connect: Vec<zenoh_config::EndPoint>,

    /// Endpoints to listen on.
    #[clap(long)]
    listen: Vec<zenoh_config::EndPoint>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Args = Args::parse();

    let window = "video capture";
    highgui::named_window_def(window)?;

    // configure zenoh
    let mut zenoh_config = Config::default();
    if !args.listen.is_empty() {
        zenoh_config.listen.endpoints = args.listen.clone();
        println!(
            "Configured listening endpoints {:?}",
            zenoh_config.listen.endpoints
        );
    }
    if !args.connect.is_empty() {
        zenoh_config.connect.endpoints = args.connect.clone();
        println!(
            "Configured connect endpoints {:?}",
            zenoh_config.connect.endpoints
        );
    }

    let zenoh_session = zenoh::open(zenoh_config)
        .res()
        .await
        .map_err(ErrorWrapper::ZenohError)?;
    let zenoh_session = zenoh_session.into_arc();

    let publisher = zenoh_session
        .declare_publisher("face-tracker/image")
        .congestion_control(CongestionControl::Drop)
        .priority(Priority::InteractiveHigh)
        .res()
        .await
        .map_err(ErrorWrapper::ZenohError)?;

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        if frame.size()?.width == 0 {
            thread::sleep(Duration::from_secs(50));
            continue;
        }

        let mut buffer: opencv::core::Vector<u8> = Default::default();

        opencv::imgcodecs::imencode_def(".jpg", &frame, &mut buffer)?;

        let data = buffer.to_vec();

        publisher
            .put(data)
            .res()
            .await
            .map_err(ErrorWrapper::ZenohError)?;

        highgui::imshow(window, &frame)?;
        _ = highgui::poll_key()?;
    }
}
