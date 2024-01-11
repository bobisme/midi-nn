use midly::num::u7;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dynamic {
    VerySoft,
    Soft,
    Medium,
    Loud,
    VeryLoud,
}

impl Dynamic {
    pub fn as_int(&self) -> u8 {
        match self {
            Self::VerySoft => 16,
            Self::Soft => 40,
            Self::Medium => 64,
            Self::Loud => 88,
            Self::VeryLoud => 112,
        }
    }

    pub fn vel(&self) -> u8 {
        self.as_int()
    }

    pub fn down(&self) -> Self {
        match self {
            Self::VerySoft => Self::VerySoft,
            Self::Soft => Self::VerySoft,
            Self::Medium => Self::Soft,
            Self::Loud => Self::Medium,
            Self::VeryLoud => Self::Loud,
        }
    }

    pub fn up(&self) -> Self {
        match self {
            Self::VerySoft => Self::Soft,
            Self::Soft => Self::Medium,
            Self::Medium => Self::Loud,
            Self::Loud => Self::VeryLoud,
            Self::VeryLoud => Self::VeryLoud,
        }
    }
}

impl From<u8> for Dynamic {
    fn from(x: u8) -> Self {
        match x {
            0..=24 => Self::VerySoft,
            25..=49 => Self::Soft,
            50..=76 => Self::Medium,
            77..=101 => Self::Loud,
            102.. => Self::VeryLoud,
        }
    }
}
impl From<u7> for Dynamic {
    fn from(x: u7) -> Self {
        Dynamic::from(x.as_int())
    }
}

impl Default for Dynamic {
    fn default() -> Self {
        Self::Medium
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Event {
    PlayNote {
        key: u7,
        dynamic: Dynamic,
    },
    PlayNoteTicks {
        key: u7,
        dynamic: Dynamic,
        ticks: u32,
    },
    StopNote {
        key: u7,
    },
    Wait {
        ticks: u32,
    },
}

impl Event {
    pub fn play(key: impl Into<u7>, dynamic: impl Into<Dynamic>) -> Self {
        Self::PlayNote {
            key: key.into(),
            dynamic: dynamic.into(),
        }
    }

    pub fn play_ticks(key: impl Into<u7>, dynamic: impl Into<Dynamic>, ticks: u32) -> Self {
        Self::PlayNoteTicks {
            key: key.into(),
            dynamic: dynamic.into(),
            ticks,
        }
    }

    pub fn stop(key: impl Into<u7>) -> Self {
        Self::StopNote { key: key.into() }
    }

    pub fn wait(ticks: impl Into<u32>) -> Self {
        Self::Wait {
            ticks: ticks.into(),
        }
    }
}
