global PATH = "H:\\OneDrive\\Forschung\\Insurance\\Data\\raw\\"
cd $PATH

import delim data_filtered_combined_s8_to_s17.csv, clear 
replace belief=belief/100

gen yprob=ln(prob/(1-prob))
gen ybel=ln(belief/(1-belief))

reg yprob var1 var2 var3 var4, noconstant
reg ybel var1 var2 var3 var4, noconstant

egen sub = group(label)
sum sub

gen b1=.
gen b2=.
gen b3=.
gen b4=.
gen cons=.

forval s=1(1)225 {
    reg ybel var1 var2 var3 var4 if sub==`s', noconstant
	replace b1 = _b[var1] if sub==`s'
	replace b2 = _b[var2] if sub==`s'
	replace b3 = _b[var3] if sub==`s'
	replace b4 = _b[var4] if sub==`s'
	*replace cons = _b[_cons] if sub==`s'
}


gen trueb1=0.7
gen trueb2=0.35
gen trueb3=-0.6
gen trueb4=-0.45
*gen truecons=0

gen diff1=b1-trueb1
gen diff2=b2-trueb2
gen diff3=b3-trueb3
gen diff4=b4-trueb4
*gen diffc=cons-truecons

save tempvb, replace

gen info1=0
replace info1=1 if treatment=="varinfo"
replace info1=1 if treatment=="posinfo"
replace info1=2 if treatment=="fullinfo"

bysort info1: sum diff1
ranksum diff1 if info1==0 | info1==1, by(info1)

gen info2=0
replace info2=1 if treatment=="varinfo"
replace info2=1 if treatment=="posinfo"
replace info2=2 if treatment=="fullinfo"

bysort info2: sum diff2
ranksum diff2 if info2==0 | info2==1, by(info2)

gen info3=0
replace info3=1 if treatment=="varinfo"
replace info3=1 if treatment=="neginfo"
replace info3=2 if treatment=="fullinfo"

bysort info3: sum diff3
ranksum diff3 if info3==0 | info3==1, by(info3)

gen info4=0
replace info4=1 if treatment=="varinfo"
replace info4=1 if treatment=="neginfo"
replace info4=2 if treatment=="fullinfo"

bysort info4: sum diff4
ranksum diff4 if info4==0 | info4==1, by(info4)

encode treatment, gen(treat)
reg diff1 i.treat timer_a round loss