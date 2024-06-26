

document.addEventListener("DOMContentLoaded", function () {
    const change = document.getElementById("change");
    
    // Use event delegation to handle both forms
    document.body.addEventListener("submit", function (event) {
        if (event.target.id === "firstform") {
            console.log('first form submit');
            event.preventDefault();
            fetch("/getcalendar", {
                method: "GET"
            }).then(response => {
                var res = response.text();
                console.log(res);
                return res;
            }).then(html => {
                change.innerHTML = html;
            });
        } else if (event.target.id === "secondform") {
            console.log('second form submit');
            event.preventDefault();
            const selected_date = document.getElementById("selected_date")
            const date = selected_date.value;
            console.log('Dato: ' + date);
            fetch("/postdate", {
                method: "POST",
                headers: {
                'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 'date': date }),
            }).then(response => {
                //pass
            }).then(() => {
                fetch('/gettime', {
                    method: 'GET',
                }).then(response => {
                    var svar = response.text();
                    return svar;
                }).then(html => {
                    change.innerHTML = html;
                });
            });
        } else if (event.target.id === "thirdform") {
            console.log('third form submit');
            event.preventDefault();
            const selected_time = document.getElementById("selected_time")
            const time = selected_time.value;
            console.log('Dato: ' + time + ":00");
            fetch('/posttime', {
                method: "POST",
                headers: {
                'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 'time': time }),
            }).then(response => {
                //pass
            }).then(() => {
                fetch('/getsolskinn', {
                    method: 'GET',
                }).then(response => {
                    var svar = response.text();
                    return svar;
                }).then(html => {
                    change.innerHTML = html;
                    const percentagePoint = document.getElementById("percentagePoint");
                    const percentageLine = document.getElementById("percentageLine");
                    const emoji = document.getElementById('emoji');

                    let isDragging = false;

                    percentagePoint.addEventListener("mousedown", (e) => {
                        isDragging = true;
                        e.preventDefault();

                        document.addEventListener("mousemove", onMouseMove);
                        document.addEventListener("mouseup", onMouseUp);
                    });

                    function onMouseMove(e) {
                        if (!isDragging) return;
        
                        const offsetX = e.clientX - percentageLine.getBoundingClientRect().left;
                        const percentage = (offsetX / percentageLine.clientWidth) * 100;
        
                        // Ensure the percentage stays within the range 0-100
                        const clampedPercentage = Math.min(Math.max(percentage, 0), 95);
        
                        percentagePoint.style.left = `${clampedPercentage}%`;

                        updateEmoji(clampedPercentage);
                    }
        
                    function onMouseUp() {
                        isDragging = false;
                        document.removeEventListener("mousemove", onMouseMove);
                        document.removeEventListener("mouseup", onMouseUp);
                    }

                    percentagePoint.style.left = "50%"; // Initial position at 50%

                    function updateEmoji(percentage) {
                        if (percentage < 33) {
                            emoji.innerHTML = "🌧️"; // Rainy cloud emoji
                        } else if (percentage >= 33 && percentage < 66) {
                            emoji.innerHTML = "🌥️"; // Cloud emoji
                        } else {
                            emoji.innerHTML = "☀️"; // Sun emoji
                        }
                    }

                });
            });
        } else if (event.target.id === "fourthform") {
            console.log('fourth form submit');
            event.preventDefault();
            const percentagePoint = document.getElementById("percentagePoint");
            const percentageLine = document.getElementById("percentageLine");
            const pointPosition = percentagePoint.offsetLeft;
            const linePosition = percentageLine.clientWidth;
            const percentage = (pointPosition / linePosition) * 100;
            
            console.log('pointpos: ' + percentage + "%");
            fetch('/postposition', {
                method: "POST",
                headers: {
                'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 'pos': percentage }),
            }).then(response => {
                //pass
            }).then(() => {
                fetch('/gettemperatur', {
                    method: 'GET',
                }).then(response => {
                    var svar = response.text();
                    return svar;
                }).then(html => {
                    change.innerHTML = html;
                    const percentagePoint = document.getElementById("percentagePoint2");
                    const percentageLine = document.getElementById("percentageLine2");
                    const tempDiv = document.getElementById("tempDisplay");
                    const tempInput = document.getElementById("temp");
                    const emoji = document.getElementById('emoji2');
                    const pointPosition = percentagePoint.offsetLeft;
                    const linePosition = percentageLine.clientWidth;
                    const percentage = (pointPosition / linePosition) * 100;

                    let isDragging = false;

                    percentagePoint.addEventListener("mousedown", (e) => {
                        isDragging = true;
                        e.preventDefault();

                        document.addEventListener("mousemove", onMouseMove);
                        document.addEventListener("mouseup", onMouseUp);
                    });

                    tempDiv.addEventListener("input", function () {
                        tempInput.value = tempDiv.textContent;
                    });

                    function onMouseMove(e) {
                        if (!isDragging) return;
        
                        const offsetX = e.clientX - percentageLine.getBoundingClientRect().left;
                        const percentage = (offsetX / percentageLine.clientWidth) * 100;

        
                        // Ensure the percentage stays within the range 0-100
                        const clampedPercentage = Math.min(Math.max(percentage, 0), 95);
        
                        percentagePoint.style.left = `${clampedPercentage}%`;

                        updateEmoji(clampedPercentage);
                        updateTemp(percentage)
                    }
        
                    function onMouseUp() {
                        isDragging = false;
                        document.removeEventListener("mousemove", onMouseMove);
                        document.removeEventListener("mouseup", onMouseUp);
                    }

                    percentagePoint.style.left = "50%"; // Initial position at 50%

                    function updateEmoji(percentage) {
                        if (percentage < 40) {
                            emoji.innerHTML = "🥶"; // Rainy cloud emoji
                        } else if (percentage >= 40 && percentage < 80) {
                            emoji.innerHTML = "😁"; // Cloud emoji
                        } else {
                            emoji.innerHTML = "🥵"; // Sun emoji
                        }
                    }

                    function updateTemp(percentage) {
                        tempDiv.innerHTML = Math.round((percentage/100 - 0.5) * 100)
                        tempInput.value = Math.round((percentage/100 - 0.5) * 100)
                    }

                });
            });
        } else if (event.target.id === "fithform") {
            console.log('fifth form submit');
            event.preventDefault();
            const temperatur = document.getElementById("temp");
            const temp = temperatur.value;
            console.log('temp: ' + temp);
            fetch("/posttemp", {
                method: "POST",
                headers: {
                'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 'temp': temp }),
            }).then(response => {
                window.location.href = "/final";
            });
        }
    });
});
