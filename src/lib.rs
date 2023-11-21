use opencv::core::{Point, Point2f, Rect, Size, Vector};
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

        if self.tracked_points.len() > 6 {
            // have tracked points
            if let Some(previous_frame_grayscale) = &self.previous_frame_grayscale {
                let new_tracked_points = track_points(
                    &frame_grayscale,
                    previous_frame_grayscale,
                    &self.tracked_points,
                )?;

                // draw on debug frame
                for feature in &new_tracked_points {
                    // convert to ints
                    let center = Point {
                        x: feature.x as i32,
                        y: feature.y as i32,
                    };
                    imgproc::circle_def(&mut debug_frame, center, 5, (0, 0, 255).into())?;
                }

                let face_rect = find_face_rectangle(&new_tracked_points)?;
                imgproc::rectangle_def(&mut debug_frame, face_rect, (0, 255, 255).into())?;

                self.tracked_points = new_tracked_points;
                self.previous_frame_grayscale = Some(frame_grayscale);
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
                // convert to ints
                let center = Point {
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
            Size {
                width: 30,
                height: 30,
            },
            Size {
                width: 0,
                height: 0,
            },
        )?;
        Ok(faces)
    }
}

pub fn find_largest_face(faces: &VectorOfRect) -> Option<Rect> {
    faces
        .into_iter()
        .max_by(|a, b| (a.height * a.width).cmp(&(b.height * b.width)))
}

pub fn find_features(image: &Mat, face: Rect) -> anyhow::Result<Vector<Point2f>> {
    let mut mask: Mat = Mat::zeros_size(image.size().unwrap(), image.typ())?.to_mat()?;
    // create mask
    imgproc::rectangle(
        &mut mask,
        face,
        (255, 255, 255).into(),
        opencv::imgproc::FILLED,
        opencv::imgproc::LineTypes::LINE_8 as i32,
        0,
    )?;

    let mut corners: Vector<Point2f> = Default::default();
    opencv::imgproc::good_features_to_track(
        image,
        &mut corners,
        200,
        0.02,
        7.0,
        &mask,
        // defaults
        3,
        false,
        0.04,
    )?;
    Ok(corners)
}

pub fn track_points(
    image: &Mat,
    previous_frame: &Mat,
    keypoints: &Vector<Point2f>,
) -> anyhow::Result<Vector<Point2f>> {
    // output moved points
    let mut moved_points: Vector<Point2f> = Default::default();
    // 1 or 0 if the point motion was detected
    let mut status: Vector<u8> = Default::default();
    // error of point motion
    let mut error_rep: Vector<f32> = Default::default();

    let criteria = opencv::core::TermCriteria {
        typ: (opencv::core::TermCriteria_EPS | opencv::core::TermCriteria_COUNT),
        max_count: 20,
        epsilon: 0.01,
    };

    opencv::video::calc_optical_flow_pyr_lk(
        previous_frame,
        image,
        keypoints,
        &mut moved_points,
        &mut status,
        &mut error_rep,
        (10, 10).into(),
        2,
        criteria,
        // defaults
        0,
        1e-4,
    )?;

    // get successfully tracked points
    // status is 0 or 1. 0 if the flow wasn't found
    let mut matched_moved_points: Vector<Point2f> = Default::default();
    for i in 0..moved_points.len() {
        if status.get(i).unwrap_or_default() == 1 {
            let error = error_rep.get(i).unwrap_or(20.0);
            if error < 5.0 {
                matched_moved_points.push(moved_points.get(i).unwrap());
            }
        }
    }

    Ok(matched_moved_points)
}

fn find_face_rectangle(keypoints: &Vector<Point2f>) -> anyhow::Result<Rect> {
    let rect = opencv::imgproc::bounding_rect(&keypoints)?;
    Ok(rect)
}

pub fn convert_to_grayscale(image: &Mat) -> anyhow::Result<Mat> {
    let mut gray: Mat = Mat::default();
    imgproc::cvt_color_def(&image, &mut gray, imgproc::COLOR_BGR2GRAY)?;
    Ok(gray)
}

pub fn jpeg_to_mat(data: &[u8]) -> anyhow::Result<Mat> {
    let buffer: Vector<u8> = opencv::core::Vector::from_slice(data);
    let frame =
        opencv::imgcodecs::imdecode(&buffer, opencv::imgcodecs::ImreadModes::IMREAD_COLOR as i32)?;
    Ok(frame)
}

pub fn mat_to_jpeg(image: &Mat) -> anyhow::Result<Vec<u8>> {
    let mut buffer: opencv::core::Vector<u8> = Default::default();
    opencv::imgcodecs::imencode_def(".jpg", &image, &mut buffer)?;
    Ok(buffer.to_vec())
}

pub struct CameraSource {
    source: opencv::videoio::VideoCapture,
}

impl CameraSource {
    pub fn new(index: i32) -> anyhow::Result<Self> {
        let camera = opencv::videoio::VideoCapture::new(index, opencv::videoio::CAP_ANY)?;
        let opened = opencv::videoio::VideoCapture::is_opened(&camera)?;
        if !opened {
            anyhow::bail!("Unable to open default camera!");
        }
        Ok(Self { source: camera })
    }

    pub fn next_reuse(&mut self, frame: &mut Mat) -> anyhow::Result<()> {
        self.source.read(frame)?;
        Ok(())
    }

    pub fn next_frame(&mut self) -> anyhow::Result<Mat> {
        let mut frame = Mat::default();
        self.next_reuse(&mut frame)?;
        Ok(frame)
    }
}

#[derive(Error, Debug)]
pub enum ErrorWrapper {
    #[error("Zenoh error {0:?}")]
    ZenohError(#[from] zenoh::Error),
}
