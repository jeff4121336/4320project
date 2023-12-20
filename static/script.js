// Default setting

// Uploaded audio 
// check /audio dir is empty or not
fetch('./audio')
    .then(response => response.json())
    .then(data => {
        const originalAudio = document.getElementById('originalAudio');
        console.log(data);
        if (data) {
            originalAudio.innerHTML = `
            <audio controls>
                <source src="/audio/${data.audio}" type="audio/wav"/>
            </audio>
            `;
        } else {
            originalAudio.innerText = 'No audio file found.';
        }
    })
    .catch(err => {
        console.error('Error:', err);
    });