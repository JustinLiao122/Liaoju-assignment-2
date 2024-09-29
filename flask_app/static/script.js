
let centroidCount = 0; 
let k = 0;

document.addEventListener('DOMContentLoaded', function() {
    const inputField = document.getElementById('kValueInput');

    inputField.addEventListener('keypress', function(event) {
        if (event.key === 'Enter') { // Check if Enter key is pressed
            const kValue = inputField.value; // Get the value from the input

            
            fetch('/set_number', {
                method: 'POST', 
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ k_value: kValue }) 
            })
            .then(response => response.json()) 
            .then(data => {
                console.log(data); 
                
                if (data.success) {
                    alert('K value set successfully!');
                    
                        fetch('/get_method')
                            .then(response => response.json())
                            .then(mData => {
                                console.log('Current k value:', mData.method);
                                method = mData.method;
                                if (method === 'Manual'){
                                    fetch('/get_k_value')
                                    .then(response => response.json())
                                    .then(kData => {
                                        console.log('Current k value:', kData.k_value);
                                        
                                        k = kData.k_value; 
                                    })
                                    .catch(error => {
                                        console.error('Error fetching k value:', error);
                                    });

                                }
                            })
                            .catch(error => {
                                console.error('Error fetching k value:', error);
                            });
                    
                
                    
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });




        }
    });
});


function displaySelected() {
    const selectedMethod = document.getElementById('methodSelect').value;

    fetch('/set_method', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ method: selectedMethod }) 
    })
    .then(response => response.json())
    .then(data => {
        console.log('Method updated to:', data.method);
        clearCentroids()
        if (data.method === 'Manual') {
            centroidCount = 0;
    
            enablePointSelection();

        }  
    })
    .catch(error => {
        console.error('Error:', error);
    });
}


function displayAnimation(animationUrl) {
    // Find or create the img or video element to display the animation
    let animationElement = document.getElementById('kmeans-animation');
    if (!animationElement) {
        // Create an img or video tag if it doesn't exist
        animationElement = document.createElement('img');
        animationElement.id = 'kmeans-animation';
        document.body.appendChild(animationElement);
    }

    // Set the animation URL to the element
    animationElement.src = animationUrl;
}






function runToConvergence() {
    
    fetch('/runtoConvergance', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('KMeans algorithm ran successfully!');
            const animationUrl = data.animation_url + '?t=' + new Date().getTime(); // Add a timestamp to force reload
            displayAnimation(animationUrl);
            clearCentroids();
        } else {
            console.error('Failed to run KMeans algorithm.');
        }
    })
    .catch(error => console.error('Error:', error));
}




// Function to enable centroid selection
function enablePointSelection() {
    const animationElement = document.getElementById('kmeans-animation');

    // Fetch the value of `k` from the server
    fetch('/get_k_value')
        .then(response => response.json())
        .then(data => {
            k = data.k_value;  // Set k based on the response from the server
            console.log(`K value is: ${k}`);
        })
        .catch(error => {
            console.error('Error fetching k value:', error);
        });

    // Add a click event listener to the plot image
    animationElement.addEventListener('click', function(event) {
        if (centroidCount >= k) {
            alert('You have already selected all centroids.');
            return; // Prevent further clicks
        }

        const rect = animationElement.getBoundingClientRect();
        const imgWidth = animationElement.clientWidth;  // Get current image width
        const imgHeight = animationElement.clientHeight; // Get current image height

        // Get the click coordinates relative to the image's position
        const clickX = event.clientX - rect.left;
        const clickY = event.clientY - rect.top;

        // Normalize the coordinates based on the image size (500px is the original size)
        const normalizedX = ((clickX ) * 0.05168)  -13.23; // Normalize to original image size
        const normalizedY = ((clickY ) * -0.06969) +13.13; // Normalize to original image size

        console.log("Click Coordinates:", clickX, clickY);
        console.log("Normalized Coordinates:", normalizedX, normalizedY);
        // Increment the count of centroids selected
        centroidCount++;

        // Send the selected point to the server as a centroid using normalized coordinates
        fetch('/add_centroid', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ x: normalizedX, y: normalizedY }) // Send the normalized coordinates
        })
        .then(response => response.json())
        .then(data => {
            console.log('Centroid added:', data.centroids); // Log selected centroids
            plotCentroid(clickX, clickY); // Call function to plot red "X" on the image at clicked position
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
}

// Function to plot the red X at the clicked location on the image
function plotCentroid(x, y) {
    const animationElement = document.getElementById('kmeans-animation');
    const plotContainer = document.querySelector('.centroid-overlay'); 
    const redX = document.createElement('div');
    redX.classList.add('red-x');
    redX.style.position = 'absolute';
    redX.style.left = `${x + animationElement.offsetLeft}px`; 
    redX.style.top = `${y + animationElement.offsetTop}px`; 
    redX.innerHTML = 'x'; 
    redX.style.color = 'red';
    redX.style.fontSize = '24px';
    redX.style.fontWeight = 'bold';
    redX.style.transform = 'translate(-50%, -50%)'; // Center the X

    // Append the red X to the plot container
    plotContainer.appendChild(redX);
}



function clearCentroids() {
    const plotContainer = document.querySelector('.centroid-overlay'); // Container for the plot
    while (plotContainer.firstChild) {
        plotContainer.removeChild(plotContainer.firstChild); // Remove all children
    }
}




function generateNewData() {
    fetch('/newData', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('New data generated!');
            const animationUrl = data.animation_url + '?t=' + new Date().getTime(); 
            displayAnimation(animationUrl);
            clearCentroids();
            centroidCount = 0;
        } else {
            console.error('Failed to generate new data');
        }
    })
    .catch(error => console.error('Error:', error));
}



function resetAlgorithm(){
    fetch('/resetAlg', {
        method:'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Reset Algorithm!');
            const animationUrl = data.animation_url + '?t=' + new Date().getTime(); 
            displayAnimation(animationUrl);
            clearCentroids();
            centroidCount = 0;
        } else {
            console.error('Failed to REset Algorithm');
        }
    })
    .catch(error => console.error('Error:', error));


}



function stepThrough() {
    fetch('/step_through', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('KMeans algorithm ran successfully!');
            clearCentroids();
            if (data.message) {
                alert(data.message);  
            }
            const animationUrl = data.animation_url + '?t=' + new Date().getTime(); // Add a timestamp to force reload
            displayAnimation(animationUrl);
        } else {
            console.error('Failed to run KMeans algorithm.');
        }
    })
    .catch(error => console.error('Error:', error));
}
