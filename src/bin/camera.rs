use face_tracker::{CameraSource, FaceTracker};
use opencv::highgui;

fn main() -> anyhow::Result<()> {
    let window = "video capture";
    highgui::named_window_def(window)?;

    let mut camera_source = CameraSource::new(0)?;

    let mut face_tracker = FaceTracker::new()?;

    loop {
        let frame = camera_source.next_frame()?;

        let debug_frame = face_tracker.process_frame(&frame)?;
        highgui::imshow(window, &debug_frame)?;
        _ = highgui::poll_key()?;
    }
}
