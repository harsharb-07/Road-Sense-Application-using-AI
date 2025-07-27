document.getElementById("trafficForm").addEventListener("submit", async function(event) {
    event.preventDefault();
    
    // Collect input data
    const startingPoint = document.getElementById("startingPoint").value;
    const destination = document.getElementById("destination").value;
    const travelTime = document.getElementById("travelTime").value;
    const dayOfWeek = document.getElementById("dayOfWeek").value;
    
    const data = {
      startingPoint: startingPoint,
      destination: destination,
      travelTime: travelTime,
      dayOfWeek: dayOfWeek
    };
    
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      const result = await response.json();
      
      // Display the prediction result
      document.getElementById("prediction-output").style.display = "block";
      document.getElementById("outputMessage").textContent = `Predicted traffic level: ${result.prediction}`;
    } catch (error) {
      console.error("Error:", error);
      document.getElementById("outputMessage").textContent = "Error in prediction. Please try again later.";
      document.getElementById("prediction-output").style.display = "block";
    }
  });
  