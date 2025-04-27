function genDigiSol() {
  var digisol = []
  errors = 0
  for (var i = 0; i < letters_per_word; i++) {
	digisol[i] = "";
	if (digi_feature) {
		digisol[i] = dict[word[i]];
		var rnum = Math.random();

		if (rnum < pe[i]) {
			errors +=1
			while (digisol[i] == dict[word[i]]) {
				digisol[i] = Math.floor(Math.random() * (999 - 100) + 100);
		    }
		}
	}
  }
  document.getElementById("digisol_errors").value+=word_num+":"+errors+";";
  return digisol
}

function sendCode(l) {
	var letter_l = document.getElementById("llid_"+l).getAttribute('dataText');
    var time = Date.now();
    var time_diff= time-word_date;
	document.getElementById("clicktimes_legend").value+=time+":"+word_num+":"+l+":"+letter_l+":"+time_diff+";";
	for (var i=0; i< letters_per_word; i++) {
		if (document.getElementById("task_card_"+i).classList.contains('highlighted')){
			document.getElementById("task_code_"+i).value=dict[letter_l];
		}
		
	}
	return 0;
}

function highlightLetter(l){
	for (var i=0; i< letters_per_word; i++) {
		document.getElementById("task_card_"+i).classList.remove('highlighted');
		document.getElementById("task_code_"+i).type="password";
	}
	document.getElementById("task_card_"+l).classList.add('highlighted');
	document.getElementById("task_code_"+l).type="text";
	var time = Date.now();
    var time_diff= time-word_date;
	document.getElementById("clicktimes_word").value+=time+":"+word_num+":"+l+":"+word[l]+":"+time_diff+";";
	return 0;
}

async function submitSolution() {
	
	document.getElementById("solution_error").style.display = "none"; 
	document.getElementById("solution_success").style.display = "none"; 	
	document.getElementById("hide_task").style.display = "none";
	for (var i=0; i< letters_per_word; i++) {
		document.getElementById("task_card_"+i).classList.remove('highlighted');
		document.getElementById("task_code_"+i).type="password";
	}
	errors=checkSolution();
	 
  if (errors == 0) {
	  var temp = Number(document.getElementById("performance").value) + 1
	  document.getElementById("performance").value = temp;
	  document.getElementById("correct_solutions").innerText = temp;
	  document.getElementById("solution_success").style.display = "block";
	  var time = Date.now();
      var time_diff= time-word_date;
      document.getElementById("clicktimes_submit").value+=time+":"+word_num+":"+time_diff+":Correct"+";";
	  await Sleep(wt_c);
  }

  else if (errors > 0) {
	  var temp = Number(document.getElementById("mistakes").value) + 1
	  document.getElementById("mistakes").value = temp;
	  document.getElementById("solution_error").style.display = "block";
	  var time = Date.now();
      var time_diff= time-word_date;
      document.getElementById("clicktimes_submit").value+=time+":"+word_num+":"+time_diff+":Incorrect"+";";
	  await Sleep(wt_i);
  }

  document.getElementById("solution_error").style.display = "none"; 
  document.getElementById("solution_success").style.display = "none"; 	
  document.getElementById("hide_task").style.display = "grid";

  
  dict=genNewDict();
  shuffle(window.legend_letters)
  for (var i = 0; i <= 25; i++) {
	document.getElementById("llid_"+i).setAttribute('dataText', legend_letters[i]);
	document.getElementById("lcid_"+i).innerText=dict[legend_letters[i]];
  }    

  word = genNewWord();
  digisol = genDigiSol();
  for (var i=0; i< letters_per_word; i++) {
	document.getElementById("task_letter_"+i).setAttribute('dataText', word[i]);
	document.getElementById("task_code_"+i).value=digisol[i];
  }
}

function Sleep(milliseconds) {
   return new Promise(resolve => setTimeout(resolve, milliseconds));
}

function checkSolution() {
	var answers = [];
	var errors = 0;
	for (var i = 0; i < letters_per_word; i++) {
		answers[i] = Number(document.getElementById("task_code_"+i).value);
		if (dict[word[i]] != answers[i]) {errors += 1}

	}
	return errors
}

function genNewWord() {
	word_num += 1;
	word_date = Date.now();
	word_date_delta =word_date - load_date;

	document.getElementById("loadtimes_word").value += word_date +":"+ word_num +":"+word_date_delta + ";";

	shuffle(word_letters)
	var word = [];
	  for (var i = 0; i < letters_per_word; i++) {
		   word[i] = word_letters[i];
	   }
	return word
}

function genCharArray(charA, charZ) {
	var a = [], i = charA.charCodeAt(0), j = charZ.charCodeAt(0);
	for (; i <= j; ++i) {
		a.push(String.fromCharCode(i));
	}
	return a;
}

function shuffle(a) {
	var j, x, i;
	var b=a;
	for (i = b.length - 1; i > 0; i--) {
		j = Math.floor(Math.random() * (i + 1));
		x = b[i];
		b[i] = b[j];
		b[j] = x;
	}
	return a;
}

function genNewDict() {
	dict = {}
	dict["A"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["B"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["C"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["D"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["E"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["F"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["G"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["H"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["I"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["J"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["K"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["L"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["M"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["N"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["O"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["P"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["Q"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["R"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["S"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["T"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["U"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["V"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["W"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["X"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["Y"] = Math.floor(Math.random() * (999 - 100) + 100);
	dict["Z"] = Math.floor(Math.random() * (999 - 100) + 100);
	shuffle(dict)
	return dict
	
}

letters_per_word = js_vars.letters_per_word;
var pe = [];
pe[0] = js_vars.pe_1;
pe[1] = js_vars.pe_2;
pe[2] = js_vars.pe_3;
pe[3] = js_vars.pe_4;
pe[4] = js_vars.pe_5;
pe[5] = js_vars.pe_6;
pe[6] = js_vars.pe_7;
pe[7] = js_vars.pe_8;

wt_c = js_vars.wt_c;
wt_i = js_vars.wt_i;

digi_feature = Boolean(js_vars.digi_feature == 1);

word_num = 0;
load_date = Date.now();

alphabet = genCharArray('A', 'Z');
legend_letters = alphabet;
word_letters = alphabet;
dict = genNewDict();
word = genNewWord();


if (document.getElementById("performance").value == "") {
	document.getElementById("performance").value = 0;
}

if (document.getElementById("mistakes").value == "") {
	document.getElementById("mistakes").value = 0;
}


dict=genNewDict();

shuffle(legend_letters)
for (var i = 0; i <= 25; i++) {
	// document.getElementById("llid_"+i).innerText=legend_letters[i];
	document.getElementById("llid_"+i).setAttribute('dataText', legend_letters[i]);
	document.getElementById("lcid_"+i).innerText=dict[legend_letters[i]];
}

digisol = genDigiSol();
for (var i=0; i< letters_per_word; i++) {
	document.getElementById("task_letter_"+i).setAttribute('dataText', word[i]);
	document.getElementById("task_code_"+i).value=digisol[i];
}

