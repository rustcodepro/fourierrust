use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "egraph",
    version = "1.0",
    about = "classification for variant annotation.
       ************************************************
       Gaurav Sablok,
       Email: codeprog@icloud.com
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Classify a single population
    Classifier {
        /// inputfile
        filepathinput: String,
        /// quality
        qualityinput: String,
        /// genotype prediction
        genotypeinput: String,
        /// thread for the analysis
        thread: String,
    },
    /// filter and classify the variants on the specific variant types for a single population
    VariantClassifier {
        /// file to be used
        fileinput: String,
        /// variant to be filtered
        variantinput: String,
        /// quality value
        qualityvalueinput: String,
        /// genotype prediction
        genotypepathinput: String,
        /// thread for the analysis
        thread: String,
    },
    /// Classify on an entire population
    Population {
        /// input directory
        directoryinput: String,
        /// qualityinput
        qualityinput: String,
        /// genotype prediction
        genotypeinput: String,
        /// thread for the analysis
        thread: String,
    },
    /// Classify an entire population over a specific variant
    PopulationVariant {
        /// input directory
        directoryinput: String,
        /// variant to be filtered
        variantinput: String,
        /// quality value input
        qualityvalue: String,
        /// genotype prediction
        genotypeprediction: String,
        /// thread for the analysis
        thread: String,
    },
}
