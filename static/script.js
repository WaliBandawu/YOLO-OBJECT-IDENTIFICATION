document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const processingStatus = document.getElementById('processingStatus');
    const results = document.getElementById('results');
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData();
        const file = fileInput.files[0];
        if (!file) return;

        formData.append('file', file);

        // Show processing status
        processingStatus.classList.remove('hidden');
        results.classList.add('hidden');
        errorMessage.classList.add('hidden');

        try {
            const response = await fetch('/process', {
                method: 'POST',
                body: formData
            });

            // Check if the response is JSON
            const contentType = response.headers.get('Content-Type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new Error('Expected JSON response but got ' + contentType);
            }

            const data = await response.json();

            if (response.ok) {
                processingStatus.classList.add('hidden');
                results.classList.remove('hidden');
                
                // Handle Image Results
                if (data.type === 'image') {
                    document.getElementById('processedImage').src = data.processedImageUrl;

                    // Display detection statistics
                    new Chart(document.getElementById('detectionChart'), {
                        type: 'bar',
                        data: {
                            labels: ['Objects Detected', 'False Positives', 'True Negatives'],
                            datasets: [{
                                data: [data.objectsDetected, data.falsePositives, data.trueNegatives],
                                backgroundColor: ['#4caf50', '#f44336', '#2196f3']
                            }]
                        }
                    });
                }

                // Handle Video Results
                if (data.type === 'video') {
                    document.getElementById('videoResult').classList.remove('hidden');
                    document.getElementById('totalFrames').textContent = data.totalFrames;
                    document.getElementById('framesWithDetections').textContent = data.framesWithDetections;
                    document.getElementById('detectionRate').textContent = data.detectionRate;

                    // Video timeline chart
                    new Chart(document.getElementById('videoTimeline'), {
                        type: 'line',
                        data: {
                            labels: data.frameNumbers,
                            datasets: [{
                                label: 'Detection Occurrences',
                                data: data.detectionOccurrences,
                                fill: false,
                                borderColor: '#2196f3'
                            }]
                        }
                    });
                }
            } else {
                throw new Error('Error processing the file. Please try again.');
            }
        } catch (error) {
            processingStatus.classList.add('hidden');
            errorMessage.classList.remove('hidden');
            errorText.textContent = error.message;
        }
    });
});
