global PATH = "H:\OneDrive\Forschung\Insurance\Data"
cd $PATH

import delim data.csv, clear
gen accuracy = prob-belief/100


graph bar accuracy, over(treatment)