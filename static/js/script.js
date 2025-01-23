
document.getElementById("predictionForm").addEventListener("submit", async function (e) {
    e.preventDefault();
    document.getElementById("output").innerText = "";
    document.getElementById("error").innerText = "";

    const fvc1 = document.getElementById("fvc1").value;
    const fvc2 = document.getElementById("fvc2").value;
    const fvc3 = document.getElementById("fvc3").value;
    const fvc4 = document.getElementById("fvc4").value;
    const fvc5 = document.getElementById("fvc5").value;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                fvc1: fvc1,
                fvc2: fvc2,
                fvc3: fvc3,
                fvc4: fvc4,
                fvc5: fvc5
            })
        });

        const result = await response.json();

        if (response.ok) {
            document.getElementById("output").innerText =
                `Predicted FVC values: ${result.prediction.join(', ')}`;
        } else {
            document.getElementById("error").innerText = result.error;
        }
    } catch (err) {
        document.getElementById("error").innerText =
            "An error occurred while making the prediction.";
    }
});
