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
                          "lutz"
                         )

new.packages <- list.of.pkg.required[!(list.of.pkg.required %in% installed.packages()[,"Package"])]

# install the packages that are not installed already
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

# add all of the packages to the library
lapply(list.of.pkg.required, library, character.only=TRUE)

################################################################
################################################################
# Functions we will use later

data.preprocessing <- function(filename, site, UTC, Lat, Long){  
    # Write what this function does...
    df1 <- amf_read_base(file = filename,
                       unzip = TRUE,
                       parse_timestamp = TRUE)
    print("File Parsed.")
  
  if("TS_1_1_1" %in% colnames(df1)){
    
    Lat <- as.numeric(Lat)
    Long <- as.numeric(Long)
    
    sharing <- readLines('/Users/johnleland/Downloads/Shared_Site_Variables.csv')
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
    
    fwrite(df1.Data.With.Posix.F, paste0("/Users/johnleland/Desktop/Ameriflux_Data/Dataset/",site,".csv"))
    fwrite(uStarTh, paste0("/Users/johnleland/Desktop/Ameriflux_Data/Dataset/USTAR_",site,".csv"))
  }else {stop("Soil Temp not present in site's data")}
}

collab <- function(filename1,filename2, i){
  # Write what this function does...

  PR <- read.csv(filename1)
  UPR <- read.csv(filename2)
  
  colnames(UPR)[which(names(UPR) == "X5.")] <- "U5"
  colnames(UPR)[which(names(UPR) == "X50.")] <- "U50"
  colnames(UPR)[which(names(UPR) == "X95.")] <- "U95"
  colnames(UPR)[which(names(UPR) == "seasonYear")] <- "YEAR"
  
  UPR1 <- UPR %>% filter(aggregationMode == "year")
  
  complete <- merge(PR,UPR1,by = "YEAR", all.x = T)
  
  str <- gsub('.csv','',filename1)
  complete <- complete %>% mutate(Site = str)
  
  fwrite(complete, paste0("/Users/johnleland/Desktop/Ameriflux_Data/Dataset/Index/",i,".csv"))
}

################################################################
################################################################

# get a list of the NEON sites
sites <- amf_sites()
df <- sites %>% filter(grepl('NEON', SITE_NAME))
print(paste0("We have found ", nrow(df), "NEON sites")

# get latitudes and longitudes for sites
lat <- as.numeric(df$LOCATION_LAT)
long <- as.numeric(df$LOCATION_LONG)

# get time zones for sites
time.zones <- tz_lookup_coords(lat,long)

# save site IDs as a vector
site.id.vec <- as.vector(df$SITE_ID)

# download AmeriFlux data for the NEON sites
print("Downloading data from AmeriFlux")
downloaded.file.list <- amf_download_base(user_id = "benjaminlucas", ### need to edit this so that it is a command line argument
                                          user_email = "ben.lucas@nau.edu", ### need to edit this so that it is a command line argument
                                          site_id = site.id.vec,
                                          data_product = "BASE-BADM",
                                          data_policy = "CCBY4.0",
                                          agree_policy = TRUE,
                                          intended_use = "model",
                                          intended_use_text = "CO2 flux modeling",
                                          out_dir = "/Users/bml438/Dropbox/nerd_stuff/NAU_Research/AmeriFlux/bl_code/data/") ### need to edit this so it uses relative paths
