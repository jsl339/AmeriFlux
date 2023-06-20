# check if the required packages are installed
list.of.pkg.required <- c("amerifluxr", # nolint
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

new.packages <- list.of.pkg.required[!(list.of.pkg.required %in% installed.packages()[, "Package"])]

# install the packages that are not installed already
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

# add all of the packages to the library
lapply(list.of.pkg.required, library, character.only=TRUE)

################################################################
################################################################
# Setting command line options
option_list <- list(
    make_option(c("-u", "--userid"), type="character", default=NULL, 
                  help="AmeriFlux account user ID", metavar="character"),
    make_option(c("-e", "--email"), type="character", default=NULL, 
                  help="AmeriFlux account email", metavar="character"),
    make_option(c("-d", "--dir"), type="character", default=NULL, 
                  help="directory to save data into", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

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
# print("Downloading data from AmeriFlux")
# downloaded.file.list <- amf_download_base(user_id = opt$userid,
#                                           user_email = opt$email,
#                                           site_id = site.id.vec,
#                                           data_product = "BASE-BADM",
#                                           data_policy = "CCBY4.0",
#                                           agree_policy = TRUE,
#                                           intended_use = "model",
#                                           intended_use_text = "CO2 flux modeling",
#                                           out_dir = opt$dir)

################################################################
################################################################
processor <- function(filename, site, UTC, Lat, Long){
  
  df1 <- amf_read_base(file = filename,
                       unzip = TRUE,
                       parse_timestamp = TRUE)
  
  print("File Parsed.")
  
  if("TS_1_1_1" %in% colnames(df1)){
    
    Lat <- as.numeric(Lat)
    Long <- as.numeric(Long)
    
    sharing <- readLines(paste0(opt$dir,"Shared_Site_Variables.csv"))
    preds <- df1 %>% select(c(sharing, starts_with("TS"), starts_with("SW")))
    
    predictors <- preds %>% select(c(1:9,starts_with("D"),starts_with("HO"),starts_with("TS_1"), 
                                     starts_with("PPFD"),starts_with("TA"),starts_with("VPD"),starts_with("NE"),
                                     starts_with("Y"),starts_with("US"),starts_with("RH"), starts_with("SW")))
    
    FC <- df1$FC
    
    
    predictors <- predictors %>% mutate(HourCos = cos(HOUR)) %>% mutate(HourSin = sin(HOUR)) %>% 
      mutate(DOYCos = cos(HOUR)) %>% mutate(DOYSin = sin(HOUR))
    
    df1 <- cbind(predictors, FC)
    
    print("Data Processed")
    
    df1 <- cbind(df1, NEE = df1$FC)
    colnames(df1)[grepl("TA_",colnames(df1))] <- "Tair"
    colnames(df1)[grepl("RH",colnames(df1))] <- "Rh"
    
    ##Calculating the VPD from RH and Tair
    
    df1.1.31.23 <- df1
    colnames(df1.1.31.23)[grepl("VPD",colnames(df1))] <- "VPD"
    colnames(df1.1.31.23)[grepl("PPFD_IN_",colnames(df1))] <- "PPFD"
    df1.1.31.23 <- cbind(df1.1.31.23, Rg = PPFD.to.Rg(df1.1.31.23$PPFD))
    
    colnames(df1.1.31.23)[grepl("USTAR",colnames(df1))] <- "Ustar"
    colnames(df1.1.31.23)[grepl("TA_",colnames(df1))] <- "Tair"
    colnames(df1.1.31.23)[grepl("RH",colnames(df1))] <- "Rh"
    
    df1.Data.With.Posix.F <- fConvertTimeToPosix(df1.1.31.23, 'YMDHM', Year = 'YEAR', Month = 'MONTH', Day = 'DAY', Hour = 'HOUR', Min = 'MINUTE')
    
    print("Converted to Posix")
    df1.Data.With.Posix.F$DateTime <- as.POSIXct(df1.Data.With.Posix.F$DateTime, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
    df1.Data.With.Posix.F <- data.frame(df1.Data.With.Posix.F)
    mins<- 15*60
    df1.Data.With.Posix.F$DateTime <- (df1.Data.With.Posix.F$DateTime + mins)
    print("Eddyproc Start")
    df1Proc.C <- sEddyProc$new(site, df1.Data.With.Posix.F, c('NEE', 'Rg', 'Tair', 'VPD', 'Ustar'))
    print("EddyProc End")
    
    off <- tz_offset(df1.Data.With.Posix.F$DateTime[1], UTC)
    
    df1Proc.C$sSetLocationInfo(LatDeg = Lat, LongDeg = Long, TimeZoneHour = off$utc_offset_h) 
    
    print("Getting Distribution")
    
    df1Proc.C$sEstUstarThresholdDistribution(nSample = 200L, probs = c(0.05, 0.5, 0.95))
    
    uStarTh <- df1Proc.C$sGetEstimatedUstarThresholdDistribution()
    
    uStar <- uStarTh %>%
      filter( aggregationMode == "year") %>%
      select( "seasonYear","uStar", "5%", "50%", "95%")
    
    fwrite(df1.Data.With.Posix.F, paste0("/Users/bml438/Desktop/data/", site, ".csv")) ############ need to change this to opt$dir
    fwrite(uStarTh, paste0("/Users/bml438/Desktop/data/", "USTAR_", site, ".csv")) ############ need to change this to opt$dir <----------------------------------------------
  }else {stop("Soil Temp variable does not exist")}
}

list.of.zipped.site.files = list.files(path="/Users/bml438/Desktop/data/", pattern="*.zip") ############ need to change this to opt$dir <----------------------------------------------

# for (i in 1:length(list.of.zipped.site.files)) {
#   tryCatch({
#     file.name = paste0("/Users/bml438/Desktop/data/", list.of.zipped.site.files[i]) ############ need to change this to opt$dir <----------------------------------------------
#     processor(file.name,site.id.vec[i],time.zones[i],lat[i],long[i])
#     print(paste0("Completed site no: ", i))
#   }, error = function(e){cat("ERROR :",conditionMessage(e), "\n")})
# }

################################################################
################################################################
collab <- function(filename1, filename2, i){
  
  filename1 <- paste0("/Users/bml438/Desktop/data/", filename1) ############ need to change this to opt$dir <----------------------------------------------
  filename2 <- paste0("/Users/bml438/Desktop/data/", filename2) ############ need to change this to opt$dir <----------------------------------------------

  PR <- read.csv(filename1)
  UPR <- read.csv(filename2)
  
  colnames(UPR)[which(names(UPR) == "X5.")] <- "U5"
  colnames(UPR)[which(names(UPR) == "X50.")] <- "U50"
  colnames(UPR)[which(names(UPR) == "X95.")] <- "U95"
  colnames(UPR)[which(names(UPR) == "seasonYear")] <- "YEAR"
  
  UPR1 <- UPR %>% filter(aggregationMode == "year")
  
  complete <- merge(PR,UPR1,by = "YEAR", all.x = TRUE)
  
  str <- gsub(".csv"," ",filename1)
  complete <- complete %>% mutate(Site = str)
  
  fwrite(complete, paste0("/Users/bml438/Desktop/data/", i, ".csv"))
}

site.csv.files <- list.files(path="/Users/bml438/Desktop/data/", pattern=glob2rx("US-*.csv")) ############ need to change this to opt$dir <----------------------------------------------

site.ustar.files <- list.files(path="/Users/bml438/Desktop/data/", pattern=glob2rx("USTAR_*.csv")) ############ need to change this to opt$dir <----------------------------------------------

for(j in 1:length(site.csv.files)){
  collab(site.csv.files[j],site.ustar.files[j],j)
  print(paste0("Done with site ", j, " of ", length(site.csv.files)))
}

################################################################
################################################################