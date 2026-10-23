use std::error::Error as StdError;
use std::fmt::{Display, Formatter};

use cutile::cuda_core::DriverError;

#[derive(Debug)]
pub enum CutileBackendError {
    Cutile(cutile::error::Error),
    Device(cutile::cuda_async::error::DeviceError),
    Driver(DriverError),
    Shape(String),
}

impl Display for CutileBackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cutile(err) => write!(f, "{err}"),
            Self::Device(err) => write!(f, "{err}"),
            Self::Driver(err) => write!(f, "{err}"),
            Self::Shape(message) => write!(f, "{message}"),
        }
    }
}

impl StdError for CutileBackendError {}

impl From<cutile::error::Error> for CutileBackendError {
    fn from(value: cutile::error::Error) -> Self {
        Self::Cutile(value)
    }
}

impl From<cutile::cuda_async::error::DeviceError> for CutileBackendError {
    fn from(value: cutile::cuda_async::error::DeviceError) -> Self {
        Self::Device(value)
    }
}

impl From<DriverError> for CutileBackendError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

