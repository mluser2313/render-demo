// Import the necessary Firebase SDK modules
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.2/firebase-app.js";
import { getDatabase, ref, onValue } from "https://www.gstatic.com/firebasejs/11.0.2/firebase-database.js";

// Firebase configuration
const firebaseConfig = {
    apiKey: "AIzaSyDYwqi50DvsjFfldyJHBFcRCb27DoFH6N0",
    authDomain: "website-test-d1fc4.firebaseapp.com",
    databaseURL: "https://website-test-d1fc4-default-rtdb.firebaseio.com",
    projectId: "website-test-d1fc4",
    storageBucket: "website-test-d1fc4.appspot.com",
    messagingSenderId: "322934018306",
    appId: "1:322934018306:web:2b2468283dedabf95f721b"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const database = getDatabase(app);

// References to AQI and pollutant data in the Realtime Database
const aqiRef = ref(database, "sensor/aqi");
const pm25Ref = ref(database, "sensor/pm25");
const pm10Ref = ref(database, "sensor/pm10");
const coRef = ref(database, "sensor/co");
const so2Ref = ref(database, "sensor/so2");

// Function to update UI elements
function updateUI(ref, elementId) {
    onValue(ref, (snapshot) => {
        const value = snapshot.val();
        document.getElementById(elementId).textContent = value !== null ? value : "No Data";
    });
}

// Update UI for each pollutant
updateUI(aqiRef, "aqi");
updateUI(pm25Ref, "pm25");
updateUI(pm10Ref, "pm10");
updateUI(coRef, "co");
updateUI(so2Ref, "so2");
