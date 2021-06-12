var cgender = "male";
var logged_in = false;
function gend( p) {
    cgender = p;
   // alert(gender);
}
function ValidateEmail(inputText) {
    var mailformat = /^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$/;
    if (inputText.value.match(mailformat)) {
        return true;
    }
    else {
        return false;
    }
}
function ValidateName(inputText) {

    var mailformat = /^[a-zA-Z]+(([',. -][a-zA-Z ])?[a-zA-Z]*)*$/;
    if (inputText.value.match(mailformat)) {
        return true;
    }
    else {
        return false;
    }
}
function ValidatePassword(pass1, pass2) {
    
    if (pass1.length < 8 || pass1 != pass2)
        return false;
    else
        return true;
}

function ValidateMob(inputText) {
    var mailformat = /^\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})$/;
    if (inputText.value.match(mailformat)) {
        return true;
    }
    else {
        return false;
    }
}
function getAge(dateString) {
    var today = new Date();
    var birthDate = new Date(dateString);
    var age = today.getFullYear() - birthDate.getFullYear();
    var m = today.getMonth() - birthDate.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) {
        age--;
    }
    return age;
}

function registerUser() {
    var inputMob = document.getElementById("mob");
    var inputEmail = document.getElementById("email");
    var inputName = document.getElementById("name");
    var checkMob = ValidateMob(inputMob);
    var checkMail = ValidateEmail(inputEmail);
    var checkName = ValidateName(inputName);
    var checkPass = ValidatePassword(document.getElementById("password1").value, document.getElementById("password2").value);
    var age = getAge(document.getElementById("bdate").value);
    var checkAge = (age < 18) ? false : true;
    


    if (checkMob && checkMail && checkName && checkPass && checkAge) {

        var usersRef = firebase.database().ref("users");
        usersRef.once('value', function (snapshot) {
            if (snapshot.hasChild(document.getElementById("mob").value)) {
                alert('exists');

            }
            else {
                usersRef.child(document.getElementById("mob").value).set({
                    name: document.getElementById("name").value,
                    bdate: document.getElementById("bdate").value,
                    gender: cgender,
                    mail: document.getElementById("email").value,
                    mobile: document.getElementById("mob").value,
                    password: document.getElementById("password1").value

                }).then(function () {
                    console.log("Document successfully written!");
                    alert("Document written successfully");
                    document.getElementById("name").value = null;
                    document.getElementById("email").value = null;
                    document.getElementById("mob").value = null;
                    document.getElementById("bdate").value = null;
                    document.getElementById("password1").value = null;
                    document.getElementById("password2").value = null;

                })
                    .catch(function (error) {
                        console.error("Error writing document: ", error);
                    });

            }
        });

    }
    else {
        if (!checkMob) {
            document.getElementById("mob").value = "";
            document.getElementById("mob").placeholder = "Invalid ! Enter again";
        }
        if (!checkMail) {
            document.getElementById("email").value = "";
            document.getElementById("email").placeholder = "Invalid ! Enter again";
        }
        if (!checkName) {
            document.getElementById("name").value = "";
            document.getElementById("name").placeholder = "Invalid ! Enter again";
        }
        if (!checkPass) {
            document.getElementById("password1").value = "";
            document.getElementById("password1").placeholder = "Invalid password or dose not matched";
            document.getElementById("password2").value = "";
            document.getElementById("password2").placeholder = "Invalid password or dose not matched";
            
        }
        if (!checkAge) {
            document.getElementById("age").innerHTML = "Sorry ! Under 18 are not allowed";
        }
    }

}

function check_password() {
document.getElementById("msg_password").innerHTML ="";
document.getElementById("msg_user").innerHTML = "";

    var user_name = document.getElementById("username").value;
    var user_pass = document.getElementById("password").value;
    console.log(user_name);
    var ref = firebase.database().ref("users");
    ref.orderByChild("mail").equalTo(user_name).once("value", function (snapshot) {
      //  console.log(snapshot.val);
        var score = snapshot.val();
        var i = 0;
        if (score != null) {
            var keys = Object.keys(score);
            // document.getElementById("but").style.visibility = "visible";

            var password = score[keys[0]].password;
            console.log(password);
            console.log(user_pass);
            if (user_pass == password) {
                console.log("euu");
                logged_in = true;
               window.location.href = "/indexhtml123";
		//window.location.href='{{ url_for( 'index_func') }}';
			
            }
            else {
                console.log("ne");
		document.getElementById("password").placeholder="Check Password and enter agian";
                 document.getElementById("msg_password").innerHTML = "Check Password and enter agian";
 	 	

            }
              //  console.log(name);
            
        }
        else {
            console.log("not");
	document.getElementById("username").placeholder="Not register";
 	document.getElementById("msg_user").innerHTML = "Sorry  ! You are not registered user";
        }
    });
}