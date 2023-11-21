use face_tracker::find_features;
use face_tracker::find_largest_face;
use face_tracker::track_points;
use face_tracker::FaceDetector;
use opencv::prelude::*;
use opencv::{core, highgui, imgproc};

fn main() -> anyhow::Result<()> {
    let window = "video capture";
    highgui::named_window_def(window)?;

    let mut frame = opencv::imgcodecs::imread_def("image.jpg")?;
    let frame_next = opencv::imgcodecs::imread_def("image2.jpg")?;
    // cam.read(&mut frame)?;
    if frame.size()?.width == 0 {
        panic!("Oh no");
    }
    let mut gray: Mat = Mat::default();
    imgproc::cvt_color_def(&frame, &mut gray, imgproc::COLOR_BGR2GRAY)?;

    let mut gray_next: Mat = Mat::default();
    imgproc::cvt_color_def(&frame_next, &mut gray_next, imgproc::COLOR_BGR2GRAY)?;

    let mut face_detector = FaceDetector::new()?;
    let faces = face_detector.detect(&gray)?;
    // let mut reduced = Mat::default();
    // imgproc::resize(
    //     &gray,
    //     &mut reduced,
    //     core::Size {
    //         width: 0,
    //         height: 0,
    //     },
    //     0.25f64,
    //     0.25f64,
    //     imgproc::INTER_LINEAR,
    // )?;
    // let mut faces = types::VectorOfRect::new();
    // face.detect_multi_scale(
    //     &reduced,
    //     &mut faces,
    //     1.1,
    //     2,
    //     objdetect::CASCADE_SCALE_IMAGE,
    //     core::Size {
    //         width: 30,
    //         height: 30,
    //     },
    //     core::Size {
    //         width: 0,
    //         height: 0,
    //     },
    // )?;

    println!("faces: {}", faces.len());
    for face in &faces {
        let scaled_face = core::Rect::new(face.x, face.y, face.width, face.height);
        imgproc::rectangle_def(&mut frame, scaled_face, (0, 255, 0).into())?;
    }

    let largest_face = find_largest_face(&faces).unwrap();

    imgproc::rectangle_def(&mut frame, largest_face, (255, 255, 0).into())?;

    // find features
    let features = find_features(&gray, largest_face)?;

    for feature in &features {
        let center = opencv::core::Point {
            x: feature.x as i32,
            y: feature.y as i32,
        };
        imgproc::circle_def(&mut frame, center, 5, (0, 0, 255).into())?;
    }

    let _res = track_points(&gray, &gray_next, &features)?;

    highgui::imshow(window, &frame)?;
    highgui::wait_key(0)?;

    Ok(())
}
