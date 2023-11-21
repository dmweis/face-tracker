use opencv::core::Point2f;
use opencv::core::Vector;
use opencv::prelude::*;
use opencv::types::VectorOfRect;
use opencv::{core, imgproc, objdetect, types};
use thiserror::Error;

pub struct FaceTracker {
    face_detector: FaceDetector,
    tracked_points: Vector<Point2f>,
    previous_frame_grayscale: Option<Mat>,
}

impl FaceTracker {
    pub fn new() -> anyhow::Result<Self> {
        let face_detector = FaceDetector::new()?;
        Ok(Self {
            face_detector,
            tracked_points: Default::default(),
            previous_frame_grayscale: None,
        })
    }

    pub fn process_frame(&mut self, frame: &Mat) -> anyhow::Result<Mat> {
        let mut debug_frame = frame.clone();
        let frame_grayscale = convert_to_grayscale(frame)?;

        if !self.tracked_points.is_empty() {
            // have tracked points
            if let Some(previous_frame_grayscale) = &self.previous_frame_grayscale {
                let new_tracked_points = track_points(
                    &frame_grayscale,
                    previous_frame_grayscale,
                    &self.tracked_points,
                )?;

                // draw on debug frame
                for feature in &new_tracked_points {
                    let center = opencv::core::Point {
                        x: feature.x as i32,
                        y: feature.y as i32,
                    };
                    imgproc::circle_def(&mut debug_frame, center, 5, (0, 0, 255).into())?;
                }
                self.tracked_points = new_tracked_points;
                return Ok(debug_frame);
            }
        }

        // no tracked points
        let detected_faces = self.face_detector.detect(&frame_grayscale)?;
        // draw on debug frame
        for face in &detected_faces {
            imgproc::rectangle_def(&mut debug_frame, face, (0, 255, 0).into())?;
        }

        let largest_face = find_largest_face(&detected_faces);
        if let Some(largest_face) = largest_face {
            let features = find_features(&frame_grayscale, largest_face)?;
            // draw on debug frame
            for feature in &features {
                let center = opencv::core::Point {
                    x: feature.x as i32,
                    y: feature.y as i32,
                };
                imgproc::circle_def(&mut debug_frame, center, 5, (0, 0, 255).into())?;
            }
            self.tracked_points = features;
            self.previous_frame_grayscale = Some(frame_grayscale);
        }

        Ok(debug_frame)
    }
}

pub struct FaceDetector {
    classifier: objdetect::CascadeClassifier,
}

impl FaceDetector {
    pub fn new() -> anyhow::Result<Self> {
        let xml = core::find_file_def("haarcascades/haarcascade_frontalface_alt.xml")?;
        let classifier = objdetect::CascadeClassifier::new(&xml)?;
        Ok(Self { classifier })
    }

    pub fn detect(&mut self, image: &Mat) -> anyhow::Result<VectorOfRect> {
        let mut faces = types::VectorOfRect::new();

        self.classifier.detect_multi_scale(
            &image,
            &mut faces,
            1.1,
            2,
            objdetect::CASCADE_SCALE_IMAGE,
            core::Size {
                width: 30,
                height: 30,
            },
            core::Size {
                width: 0,
                height: 0,
            },
        )?;
        Ok(faces)
    }
}

pub fn find_largest_face(faces: &VectorOfRect) -> Option<opencv::core::Rect> {
    faces
        .into_iter()
        .max_by(|a, b| (a.height * a.width).cmp(&(b.height * b.width)))
}

pub fn find_features(
    image: &Mat,
    face: opencv::core::Rect,
) -> anyhow::Result<opencv::core::Vector<Point2f>> {
    let mut mask: Mat = Mat::zeros_size(image.size().unwrap(), image.typ())?.to_mat()?;
    imgproc::rectangle(
        &mut mask,
        face,
        (255, 255, 255).into(),
        opencv::imgproc::FILLED,
        opencv::imgproc::LineTypes::LINE_8 as i32,
        0,
    )?;

    let mut corners: opencv::core::Vector<Point2f> = Default::default();
    opencv::imgproc::good_features_to_track(
        image,
        &mut corners,
        1000,
        0.02,
        7.0,
        &mask,
        3,
        false,
        0.04,
    )?;
    Ok(corners)
}

pub fn track_points(
    frame: &Mat,
    previous_frame: &Mat,
    keypoints: &opencv::core::Vector<Point2f>,
) -> anyhow::Result<opencv::core::Vector<Point2f>> {
    // output moved points
    let mut moved_points: opencv::core::Vector<Point2f> = Default::default();
    // 1 or 0 if the point motion was detected
    let mut status: opencv::core::Vector<u8> = Default::default();
    // error of point motion
    let mut error_rep: opencv::core::Vector<f32> = Default::default();

    let criteria = opencv::core::TermCriteria {
        typ: (opencv::core::TermCriteria_EPS | opencv::core::TermCriteria_COUNT),
        max_count: 20,
        epsilon: 0.01,
    };

    opencv::video::calc_optical_flow_pyr_lk(
        previous_frame,
        frame,
        keypoints,
        &mut moved_points,
        &mut status,
        &mut error_rep,
        (10, 10).into(),
        2,
        criteria,
        0,
        1e-4,
    )?;

    // get successfully tracked points
    // status is 0 or 1. 0 if the flow wasn't found

    let mut matched_moved_points: opencv::core::Vector<Point2f> = Default::default();

    for i in 0..moved_points.len() {
        if status.get(i).unwrap_or_default() == 1 {
            matched_moved_points.push(moved_points.get(i).unwrap());
        }
    }

    println!("{:?}", error_rep);

    Ok(matched_moved_points)
}

pub fn convert_to_grayscale(image: &Mat) -> anyhow::Result<Mat> {
    let mut gray: Mat = Mat::default();
    imgproc::cvt_color_def(&image, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    Ok(gray)
}

#[derive(Error, Debug)]
pub enum ErrorWrapper {
    #[error("Zenoh error {0:?}")]
    ZenohError(#[from] zenoh::Error),
}
