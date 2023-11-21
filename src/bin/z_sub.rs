use clap::Parser;
use face_tracker::{ErrorWrapper, FaceTracker};
use opencv::highgui;
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

    let subscriber = zenoh_session
        .declare_subscriber("face-tracker/image")
        .best_effort()
        .res()
        .await
        .map_err(ErrorWrapper::ZenohError)?;

    let mut face_tracker = FaceTracker::new()?;

    loop {
        let msg = subscriber.recv_async().await?;
        let payload: Vec<u8> = msg.value.try_into()?;

        let buffer: opencv::core::Vector<u8> = opencv::core::Vector::from_slice(&payload);
        let frame = opencv::imgcodecs::imdecode(
            &buffer,
            opencv::imgcodecs::ImreadModes::IMREAD_COLOR as i32,
        )?;

        let debug_frame = face_tracker.process_frame(&frame)?;
        highgui::imshow(window, &debug_frame)?;
        _ = highgui::poll_key()?;
    }
}
