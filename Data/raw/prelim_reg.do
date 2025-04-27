global PATH = "H:\\OneDrive\\Forschung\\Insurance\\Data\\raw\\"
cd $PATH

import delim data_combined_s8_to_s17.csv, clear 
encode treatment, gen(treat)
egen sub=group(label)


replace belief=belief/100
gen diff = belief - prob
gen absdiff =abs(diff)

* Timers for subjects who did not look at the information is recorded as missing. 
foreach v in timer_char1 timer_char2 timer_char3 timer_char4 timer_char7 {
	replace `v' = 0 if `v'==.
	replace `v'=`v'/1000
}




* For five observations some timers were not recorded correctly (contain numbers > 10^10). We replace these values with the corresponding average of the same subject.
sum timer_char7 if sub==107 & timer_char7 < 10000000
replace timer_char7 = `r(mean)' if sub==107 & timer_char7 > 10000000

foreach s in 95 96 127 138 {
	sum timer_char1 if sub==`s' & timer_char1 < 10000000
	replace timer_char1 = `r(mean)' if sub==`s' & timer_char1 > 10000000
}

gen timer_all = timer_char1 + timer_char2 + timer_char3 + timer_char4 + timer_char7
egen timer_rank=rank(timer_all)
gen noread=timer_all==0



* Absolute differences between belief and true probability lower in FullInfo and higher in VarInfo
bysort treat: sum diff absdiff
reg absdiff i.treat, cluster(sub)

* Loss amount and risk measure do not affect accuracy. There are no time trends.
reg absdiff i.treat loss_amount number_entered round_number, cluster(sub)



* Checking for the impact of time spent on reading information

* Subjects did look at the information
bysort treat: sum timer_all noread
hist timer_all, by(treat)

* Those who spend more time looking at the information have slightly more accurate beliefs.
reg absdiff timer_all, cluster(sub)
reg absdiff timer_rank, cluster(sub)
reg absdiff i.treat timer_rank timer_all, cluster(sub)


* A closer look reveals some insights. Reading FullInformation was particularly helpful. Binary variables are more helpful than continuous ones. Subjects in VarInfo are just plainly confused.
bysort treat: reg absdiff timer_all, cluster(sub)
bysort treat: reg absdiff timer_char*, cluster(sub)
reg absdiff timer_char* if treat==1, cluster(sub)
reg absdiff timer_char7 if treat==2, cluster(sub)
reg absdiff timer_char3 timer_char4 if treat==3, cluster(sub)
reg absdiff timer_char1 timer_char2 if treat==4, cluster(sub)
reg absdiff timer_char1 timer_char2 timer_char3 timer_char4 if treat==5, cluster(sub)

* Trying to recover the beta coefficients from the data

gen yprob=ln(prob/(1-prob))
gen ybel=ln(belief/(1-belief))

reg yprob var1 var2 var3 var4, noconstant
reg ybel var1 var2 var3 var4, noconstant



***
*drop eps test
gen eps = rnormal(0,.5)
gen test = prob - eps
replace test=ln(test/(1-test))

reg test var1 var2 var3 var4, noconstant


gen b1=.
gen b2=.
gen b3=.
gen b4=.
gen cons=.

gen tb1=.
gen tb2=.
gen tb3=.
gen tb4=.
gen tcons=.


*replace var1=var1/10
*replace var3=var3/10

forval s=1(1)225 {
	qui{
    reg ybel var1 var2 var3 var4 if sub==`s', noconstant
	replace b1 = _b[var1] if sub==`s'
	replace b2 = _b[var2] if sub==`s'
	replace b3 = _b[var3] if sub==`s'
	replace b4 = _b[var4] if sub==`s'
	*replace cons = _b[_cons] if sub==`s'
	
    reg yprob var1 var2 var3 var4 if sub==`s', noconstant
	replace tb1 = _b[var1] if sub==`s'
	replace tb2 = _b[var2] if sub==`s'
	replace tb3 = _b[var3] if sub==`s'
	replace tb4 = _b[var4] if sub==`s'
	*replace tcons = _b[_cons] if sub==`s'
	}
}

* True betas are recovered at the individual level when using the true probability
forval i=1(1)4{
	gen bdiff`i' = tb`i'-b`i'
}

* Betas recovered from beliefs are quite removed from the predictions. It seems like the binary variables are better reflected by the beliefs, but that is probably to be expected.
sum tb* b1-b4 
bysort treat: sum bdiff*
*hist bdiff4, by(treat)
cdfplot wtp, by(treat)

gen exploss=prob*loss
graph twoway scatter wtp exploss, by(treat)

reg wtp i.treat loss belief prob number round_number var1 var2 var3 var4, cluster(sub)
reg wtp i.treat loss  number round_number var1 var2 var3 var4, cluster(sub)
reg wtp i.treat loss belief number round_number var1 var2 var3 var4 timer_all, cluster(sub)

drop test
gen test=wtp - (1*loss )
gen test2=wtp - (prob*loss )
bysort treat: sum test
reg test i.treat round number_entered, cluster(sub)
reg test i.treat round number_entered timer_all, cluster(sub)
reg test i.treat round number_entered timer_rank, cluster(sub)
reg wtp loss belief i.treat round, r


gen test2 = hllf_s -hl_s
bysort gender: sum test2
reg test ib2.treat round , cluster(sub)

bysort treat: reg test hl_s, cl(sub)

gen foo = belief*loss
gen bar = prob*loss

graph twoway scatter wtp bar || function y=x, by(treat) range(0 100)


