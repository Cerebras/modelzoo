var globalHDF5Data = {};
var specialCharToRemove;

var mode;
var globalDPOMode;

// For showing sequence distribution progress.
document.getElementById('spinner').style.display = 'block';
document.getElementById('sequence-image').style.display = 'none';

function fetchDataParams() {
    fetch('/get_data_params')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            mode = data.setup.mode;
        })
        .catch(error => {
            console.error('Unble to read data_params.json file!', error);
        });
}

function updateDisplayContent(inputIds, labelIds, inputStrings, labelStrings, dpoMode) {
    function escapeHtml(text) {
        var map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return String(text).replace(/[&<>"']/g, function(m) { return map[m]; });
    }

    const createSpan = (data, className, joinChar=' ') => {
        return data.map((num, j) => {
            const escapedStr = escapeHtml(num);
            return `<span id="${j}" class="transition-all ${className}">${escapedStr}</span>`;
        }).join(joinChar);
    };

    const createSpanWithBG = (data, className) => {
        return data.map((str, j) => {
            const escapedStr = escapeHtml(str);
            var attnWeight = dpoMode == 'C' ? globalHDF5Data.chosen_attention_mask[j] : dpoMode == 'R'? globalHDF5Data.rejected_attention_mask[j] : globalHDF5Data.attention_mask[j];
            var lossWeight;

            if ('chosen_loss_mask' in globalHDF5Data && dpoMode == 'C') {
                lossWeight = globalHDF5Data.chosen_loss_mask[j];
            } else if ('rejected_loss_mask' in globalHDF5Data && dpoMode == 'R') {
                lossWeight = globalHDF5Data.rejected_loss_mask[j];
            } else if ('loss_mask' in globalHDF5Data) {
                lossWeight = globalHDF5Data.loss_mask[j];
            } else {
                // Fallback if none of the masks exist
                lossWeight = 1 - attnWeight;
            }

            return `<span id="${j}" class="transition-all ${className}" style="background-color: ${getGreenOpacity(lossWeight, attnWeight, className)}">${escapedStr}</span>`;
        }).join(' ');
    };

    document.getElementById('input_ids_display').innerHTML = createSpan(inputIds, 'input_ids', ', ');
    document.getElementById('label_ids_display').innerHTML = createSpan(labelIds, 'label_ids', ', ');
    document.getElementById('input_strings_display').innerHTML = createSpanWithBG(inputStrings, 'input_strings');
    document.getElementById('label_strings_display').innerHTML = createSpanWithBG(labelStrings, 'label_strings');
}


function escapeSpecialCharacters(str) {
    return str
        .replace(/\\/g, '\\\\')  // Escape backslashes
        .replace(/\n/g, '\\n')   // Escape newlines
        .replace(/\t/g, '\\t')   // Escape tabs
}

function removeSpecialCharacter() {
    const inputField = document.getElementById('special-characters');
    specialCharToRemove = inputField.value
    loadData(0, specialCharToRemove, true);
}

function loadData(sequence, special_char, file=false) {
    // Get data_params in a JSON structure.
    fetchDataParams();

    const filename = document.getElementById('fileSelector').value;
    const loader = document.getElementById('loader');
    const content = document.getElementById('content');

    // Show the loader and blur the content
    loader.style.display = 'flex';
    content.classList.add('blur');

    fetch('/data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `filename=${encodeURIComponent(filename)}&sequence=${encodeURIComponent(sequence)}`
    })
    .then(response => response.json())
    .then(data => {
        if(mode == 'dpo'){
            const [sequenceNo, dpoMode] = document.getElementById('stringLengthSelector').value.split(',');
            globalDPOMode = dpoMode;

            data.chosen_input_strings = data.chosen_input_strings.map(escapeSpecialCharacters);
            data.rejected_input_strings = data.rejected_input_strings.map(escapeSpecialCharacters);

            if(special_char){
                const removeSpecialCharacters = str => str.replace(new RegExp(special_char, 'g'), '');
                // Remove special characters from input strings.
                data.chosen_input_strings = data.chosen_input_strings.map(str => removeSpecialCharacters(str, special_char));
                data.rejected_input_strings = data.rejected_input_strings.map(str => removeSpecialCharacters(str, special_char));

                // Remove special characters from label strings.
                data.chosen_label_strings = data.chosen_label_strings.map(str => removeSpecialCharacters(str, special_char));
                data.rejected_label_strings = data.rejected_label_strings.map(str => removeSpecialCharacters(str, special_char));
            }

            globalHDF5Data = data;
            updateDisplay(globalHDF5Data);
            
            if(file){
                updateStringOptions(globalHDF5Data.nstrings);
            }
            updateSections(dpoMode);

            // Hide the loader and remove the blur
            loader.style.display = 'none';
            content.classList.remove('blur');

        } else {
            dpoMode = null;
            data.input_strings = data.input_strings.map(escapeSpecialCharacters);

            if(special_char){
                const removeSpecialCharacters = str => str.replace(new RegExp(special_char, 'g'), '');
                data.input_strings = data.input_strings.map(str => removeSpecialCharacters(str, special_char)); 
                data.label_strings = data.label_strings.map(str => removeSpecialCharacters(str, special_char));
            }

            globalHDF5Data = data;
            updateDisplay(globalHDF5Data);
            if(file) updateStringOptions(globalHDF5Data.nstrings);
            updateSections(dpoMode);
            
            loader.style.display = 'none';
            content.classList.remove('blur');
        }
    })
    .catch(error => {
        console.error('Error fetching data:', error);
        // Hide the loader and remove the blur in case of an error
        loader.style.display = 'none';
        content.classList.remove('blur');
    });
}

function updateDisplay(data) {
    // Fetch sequence distribtion.
    fetch('/generate_sequence_distribution', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.image_path) {
            // Update the image source with the generated image
            const sequenceDistDiv = document.querySelector('.sequence-distribution');
            let sequenceImage = sequenceDistDiv.querySelector('img');

            if(!sequenceImage){
                sequenceImage = document.createElement('img')
                sequenceImage.id = 'sequence-image';
                sequenceDistDiv.appendChild(sequenceImage);
            }
            sequenceImage.src = data.image_path;

            sequenceImage.onload = () => {
                document.getElementById('spinner').style.display = 'none';
                sequenceImage.style.display = 'block';
            };

            sequenceImage.addEventListener('click', () => {
                sequenceImage.classList.toggle('zoomed'); // Toggle zoom class
                const overlay = document.querySelector('.overlay');
                if (sequenceImage.classList.contains('zoomed')) {
                    overlay.style.display = 'block'; // Show overlay
                } else {
                    overlay.style.display = 'none'; // Hide overlay
                }
            });

            const overlay = document.querySelector('.overlay');
            overlay.addEventListener('click', () => {
                sequenceImage.classList.remove('zoomed'); // Remove zoom
                overlay.style.display = 'none'; // Hide overlay
            });

            sequenceImage.onerror = () => {
                document.getElementById('spinner').style.display = 'none';
                console.error('Error loading the image');
            };
        } else {
            console.error('Image path not provided in the response');
        }
    })
    .catch(error => {
        console.error('Error generating sequence distribution image:', error);
    });  

    // Updating stats table
    const statsTable = document.getElementById('statsTable');
    statsTable.innerHTML = '';
    Object.entries(data.stats).forEach(([stat, value], index, array) => {
        const row = statsTable.insertRow();
        const cell1 = row.insertCell(0);
        const cell2 = row.insertCell(1);
        cell1.textContent = stat;
        cell2.textContent = JSON.stringify(value, null, 4);
    
        // Add a separator row unless it's the last row
        if (index < array.length - 1) {
            const separatorRow = statsTable.insertRow();
            const separatorCell = separatorRow.insertCell(0);
            separatorCell.colSpan = 2;
            separatorCell.className = 'separator-row';
        }
    });          
}

function updateStringOptions(nstrings) {
    var stringLengthSelector = document.getElementById('stringLengthSelector');
    stringLengthSelector.innerHTML = ''; // Clear existing options    
    // Populate dropdown based on the number of strings
    if(mode == 'dpo'){
        for (let i = 1; i <= (nstrings/2); i++){
            var option_chosen = document.createElement('option')
            var option_rejected = document.createElement('option')

            option_chosen.value = `${i},C`;
            option_rejected.value = `${i},R`;

            option_chosen.text = `${i},C`;
            option_rejected.text = `${i},R`;

            stringLengthSelector.appendChild(option_chosen);
            stringLengthSelector.appendChild(option_rejected);
        }
    }
    else{
        for (let i = 1; i <= nstrings; i++) {
            var option = document.createElement('option');
            option.value = i;
            option.text = i;
            stringLengthSelector.appendChild(option);
        }
    }
};

function updateSections(dpoMode = 'C') {
    if (mode == 'dpo'){
        if (dpoMode == 'C'){
            updateDisplayContent(globalHDF5Data.chosen_input_ids, globalHDF5Data.chosen_labels, globalHDF5Data.chosen_input_strings, globalHDF5Data.chosen_label_strings, dpoMode);
        }
        else if(dpoMode == 'R'){
            updateDisplayContent(globalHDF5Data.rejected_input_ids, globalHDF5Data.rejected_labels, globalHDF5Data.rejected_input_strings, globalHDF5Data.rejected_label_strings, dpoMode);
        }
    } else {
        updateDisplayContent(globalHDF5Data.input_ids, globalHDF5Data.labels, globalHDF5Data.input_strings, globalHDF5Data.label_strings, dpoMode);
    }

    registerHoverEvents();
}
        
document.getElementById('stringLengthSelector').addEventListener('change', function() {
    if(mode == 'dpo'){
        const [sequenceNo, dpoMode] = this.value.split(',')
        loadData(sequenceNo - 1, specialCharToRemove);
    } else {
        loadData(this.value - 1, specialCharToRemove);
    }   
});


function isElementInViewport(el, container) {
    const rect = el.getBoundingClientRect();
    const containerRect = container.getBoundingClientRect();

    return (
        rect.top >= containerRect.top &&
        rect.left >= containerRect.left &&
        rect.bottom <= containerRect.bottom &&
        rect.right <= containerRect.right
    );
}

function scrollIntoViewIfNeeded(el, container) {
    if (!isElementInViewport(el, container)) {
        container.scrollTop = el.offsetTop - container.offsetTop - container.clientHeight/2 + el.clientHeight/2 - 25;
    }
}

function syncScroll(event) {
    const target = event.target;
    const scrollTop = target.scrollTop;
    const scrollLeft = target.scrollLeft;

    document.querySelectorAll('.data-display').forEach(element => {
        if (element.id == 'label_strings_display') {
            element.scrollTop = scrollTop;
            element.scrollLeft = scrollLeft;
        }
    });
}

// Add scroll event listener to each data-display section
document.querySelector('#input_strings_display').addEventListener('scroll', syncScroll);

// Dynamically resize the stats table to avoid underflow/overflow.
window.addEventListener('resize', () => {
    const statsTable = document.getElementById('statsTable');
    const container = statsTable.parentElement;

    // Adjust table width
    statsTable.style.width = `${container.clientWidth}px`;

    // Adjust font size based on window width
    const newFontSize = Math.max(12, Math.min(16, container.clientWidth / 50));
    statsTable.style.fontSize = `${newFontSize}px`;

    // Adjust row height based on font size
    const rows = statsTable.getElementsByTagName('tr');
    for (let row of rows) {
        row.style.height = `${newFontSize * 1.5}px`;
    }
});
