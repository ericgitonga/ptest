# Contents of the .env file

## EarthRanger
ER_KB_SERVER=
ER_KB_USERNAME=
ER_KB_PASSWORD=

## Directories
OUTPUT_DIR=Ecoscope-Outputs/KBOPT/

## Time
START=2024-01-01
END=2024-06-30

## KBoPT
KB_CONSERVANCIES={"lc":"LC_","mmnr":"MMNR_","mnc":"MNC_","mt":"MT_","nc":"NC_","oc":"OC_","okc":"OKC_","omc":"OMC_","pca":"PCA_","sc":"SC_","snp":"SNP_","nca":"NCA_"}
                  
## EarthRanger Event Types
ER_NEST_ID_EVENT=nest_id
ER_NEST_CHECK_EVENT=nest_check

## Column definitions
NC_COLS=["nest_id","observer","date","time","latitude","longitude","method","status","species","condition","incubating","nest_outcome","adult_id_male","picture_taken","age_of_chick_1","age_of_chick_2","age_of_chick_3","age_of_chick_4","number_of_eggs","adult_id_female","number_of_chicks","number_of_fledglings","eggs_chicks_fledglings","number_of_adults_present","breeding_attempt","comments","serial_number"]

NEST_CHECK_COLUMNS=["nest_id","date","species","status","latitude","longitude","number_of_eggs","number_of_chicks","number_of_fledglings","incubating","altitude","habitat","tree_species","tree_cbh","tree_or_cliff_height","height","position","observer","time","nest_location","condition","checks_serial_number"]

INACTIVE_COLUMNS=["nest_id","observer","nest_location","date","time","latitude","longitude"]

SUCCESS_FAIL_COLUMNS=["nest_id","date","species","status","latitude","longitude","altitude","habitat","tree_species","tree_or_cliff_height","height","position","condition"]

IN_PROGRESS_COLUMNS=["nest_id","date","species","status","number_of_eggs","number_of_chicks","incubating","latitude","longitude","altitude","habitat","tree_species","tree_cbh","tree_or_cliff_height","height","position","condition"]

NEST_ID_COLUMNS=["serial_number","nest_id","nest_location","altitude","habitat","tree_species","tree_cbh","tree_or_cliff_height","height","position"]
