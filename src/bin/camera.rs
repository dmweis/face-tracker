use face_tracker::FaceTracker;
use opencv::highgui;
use opencv::prelude::*;
use std::thread;
use std::time::Duration;

use opencv::videoio;

fn main() -> anyhow::Result<()> {
    let window = "video capture";
    highgui::named_window_def(window)?;

    let mut cam = videoio::VideoCapture::new(0, videoio::CAP_ANY)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }

    let mut face_tracker = FaceTracker::new()?;

    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        if frame.size()?.width == 0 {
            thread::sleep(Duration::from_secs(50));
            continue;
        }

        let debug_frame = face_tracker.process_frame(&frame)?;
        highgui::imshow(window, &debug_frame)?;
    }
}
