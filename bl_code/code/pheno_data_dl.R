# check if the required packages are installed
list.of.pkg.required <- c("optparse",
                          "devtools"
                         )

new.packages <- list.of.pkg.required[!(list.of.pkg.required %in% installed.packages()[, "Package"])]

# install the packages that are not installed already
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

# add all of the packages to the library
lapply(list.of.pkg.required, library, character.only=TRUE)

################################################################
################################################################
# Setting command line options
option_list <- list(
    make_option(c("-d", "--dir"), type="character", default=NULL, 
                  help="directory to save data into", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

################################################################
################################################################

# installing the package from the GitHub repo
if(!require(devtools)) install.packages('devtools')
devtools::install_github('bnasr/phenocamapi')

# loading the package
library(phenocamapi)

# Get site metadata and rois
phenos <- get_phenos()
#For some reason, when run in succession, it might fail at get_rois
#But can be rerun from this line on and seems to work fine. 
rois <- get_rois()

#List of neon site names
neon_sites <- c('PR-xGU',
                'PR-xLA',
                'US-xAB',
                'US-xBA',
                'US-xBL',
                'US-xBN',
                'US-xBR',
                'US-xCL',
                'US-xCP',
                'US-xDC',
                'US-xDJ',
                'US-xDL',
                'US-xDS',
                'US-xGR',
                'US-xHA',
                'US-xHE',
                'US-xJE',
                'US-xJR',
                'US-xKA',
                'US-xKZ',
                'US-xLE',
                'US-xMB',
                'US-xNG',
                'US-xNQ',
                'US-xNW',
                'US-xRM',
                'US-xRN',
                'US-xSB',
                'US-xSC',
                'US-xSE',
                'US-xSJ',
                'US-xSL',
                'US-xSP',
                'US-xSR',
                'US-xST',
                'US-xTA',
                'US-xTE',
                'US-xTL',
                'US-xTR',
                'US-xUK',
                'US-xUN',
                'US-xWD',
                'US-xWR',
                'US-xYE')

#Rename mislabeled site in phenos
phenos$flux_sitenames[438] <- 'US-xSL'
phenos$flux_sitenames[439] <- 'US-xSL'

#Filter phenos by flux sitename in neon_sites
neon_phenos <- phenos[phenos$flux_sitenames %in% neon_sites,]

#Take first instance of each site name 
filtered_neon_phenos <- neon_phenos[!duplicated(neon_phenos$flux_sitenames),]
#Alphabetize by site name 
filtered_neon_phenos <- filtered_neon_phenos[order(filtered_neon_phenos$flux_sitenames)]

#Filter ROI list by site name
neon_site_names <- strsplit(filtered_neon_phenos$site, " ")
neon_rois <- rois[rois$site %in% neon_site_names,]
#Take first instance of sitename when multiple vegtypes exist
neon_rois <- neon_rois[!duplicated(neon_rois$site),]
#Match to alphabetized order
neon_rois <- neon_rois[match(filtered_neon_phenos$site,neon_rois$site)]
#Add column with site name abbreviations
neon_rois$abrev <- filtered_neon_phenos$flux_sitenames

# Read 1 day data for each neon site and concatenate
for (i in 1:nrow(neon_rois)){
  input1 <- neon_rois$site[i]
  input2 <- neon_rois$roitype[i]
  input3 <- neon_rois$sequence_number[i]
  name <- neon_rois$abrev[i]
       
  gcc <- get_pheno_ts(input1, input2, input3, '1day')
  gcc$site <- name
  if(exists('full_df')){
    full_df <-rbind(full_df,gcc)
           }
  else{
    full_df <- gcc
           }
}

fwrite(full_df, paste0("/Users/bml438/Desktop/data/", "all_sites_phenocam.csv"))  ############ need to change this to opt$dir <----------------------------------------------