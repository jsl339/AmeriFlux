# check if the required packages are installed
list.of.pkg.required <- c("amerifluxr",
                          "REddyProc",
                          "dplyr",
                          "data.table",
                          "bigleaf",
                          "zoo",
                          "plyr",
                          "tidyverse",
                          "imputeTS",
                          "lutz",
                          "optparse"
                         )

new.packages <- list.of.pkg.required[!(list.of.pkg.required %in% installed.packages()[,"Package"])]

# install the packages that are not installed already
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

# add all of the packages to the library
lapply(list.of.pkg.required, library, character.only=TRUE)

################################################################
################################################################
# Setting command line options
option_list = list(
    make_option(c("-u", "--userid"), type="character", default=NULL, 
                  help="AmeriFlux account user ID", metavar="character"),
    make_option(c("-e", "--email"), type="character", default=NULL, 
                  help="AmeriFlux account email", metavar="character"),
    make_option(c("-d", "--dir"), type="character", default=NULL, 
                  help="directory to save data into", metavar="character")
)
 
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

################################################################
################################################################
# get a list of the NEON sites
sites <- amf_sites()
df <- sites %>% filter(grepl('NEON', SITE_NAME))
print(paste0("We have found ", nrow(df), " NEON sites"))

# get latitudes and longitudes for sites
lat <- as.numeric(df$LOCATION_LAT)
long <- as.numeric(df$LOCATION_LONG)

# get time zones for sites
time.zones <- tz_lookup_coords(lat,long)

# save site IDs as a vector
site.id.vec <- as.vector(df$SITE_ID)

# download AmeriFlux data for the NEON sites
print("Downloading data from AmeriFlux")
downloaded.file.list <- amf_download_base(user_id = opt$userid,
                                          user_email = opt$email,
                                          site_id = site.id.vec,
                                          data_product = "BASE-BADM",
                                          data_policy = "CCBY4.0",
                                          agree_policy = TRUE,
                                          intended_use = "model",
                                          intended_use_text = "CO2 flux modeling",
                                          out_dir = opt$dir) ### need to edit this to opt$dir!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

################################################################
################################################################



################################################################
################################################################