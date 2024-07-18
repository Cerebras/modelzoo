var globalHDF5Data = {};
document.addEventListener('DOMContentLoaded', function() {
    const controlsSection = document.querySelector('.controls-section');
    const controls = document.querySelector('.controls');
    const dynamicSections = document.querySelector('.dynamic-sections');
    const controlsInitialTop = controlsSection.offsetTop;
    controls.style.width = window.getComputedStyle(dynamicSections).getPropertyValue('width');
    window.addEventListener('scroll', function() {
        if (window.scrollY > controlsInitialTop + 100) {
            controlsSection.classList.add('fixed-controls');
        } else {
            controlsSection.classList.remove('fixed-controls');
        }
    });
});

function loadData(sequence, file=false) {
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
        globalHDF5Data = data;
        updateDisplay(globalHDF5Data);
        if(file) updateStringOptions(globalHDF5Data.nstrings);
        updateSections();
        // Hide the loader and remove the blur
        loader.style.display = 'none';
        content.classList.remove('blur');
    })
    .catch(error => {
        console.error('Error fetching data:', error);

        // Hide the loader and remove the blur in case of an error
        loader.style.display = 'none';
        content.classList.remove('blur');
    });
}

function updateDisplay(data) {
    const sortedTokens = Object.entries(data.top_tokens).sort((a, b) => b[1] - a[1]);
    const tokensList = document.getElementById('tokensList');
    tokensList.innerHTML = '';
    const maxCount = sortedTokens[0][1]; // Get the maximum count for scaling bars
    // const maxWidth = 300; // Maximum width of a bar in pixels
    const availableWidth = document.getElementsByClassName('token-list')[0].clientWidth;
    const maxWidth = 0.55*availableWidth; // Set maxWidth to 50px less than the available width

    sortedTokens.forEach(([token, count]) => {
        const li = document.createElement('li');
        li.className = 'token-item'; // Add a class for styling
        const label = document.createElement('span');
        if(token === '\n')
            label.textContent = '"\\n"';
        else
            label.textContent = `${token} `;
        label.className = 'token-label';

        const barContainer = document.createElement('div');
        barContainer.className = 'bar-container';

        const bar = document.createElement('div');
        bar.className = 'token-bar';
        const calculatedWidth = Math.max(10, (count / maxCount) * maxWidth); // Calculate pixel width
        bar.style.width = `${calculatedWidth}px`; // Apply calculated width in pixels

        const countSpan = document.createElement('span');
        countSpan.textContent = count;
        countSpan.className = 'token-count';

        barContainer.appendChild(bar);
        barContainer.appendChild(countSpan);
        li.appendChild(label);
        li.appendChild(barContainer);
        tokensList.appendChild(li);
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
    for (let i = 1; i <= nstrings; i++) {
        var option = document.createElement('option');
        option.value = i;
        option.text = i;
        stringLengthSelector.appendChild(option);
    }
};

function updateSections() {
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
            var lossWeight;
            const attnWeight = globalHDF5Data.attention_mask[j];
            if('loss_mask' in globalHDF5Data)
                lossWeight = globalHDF5Data.loss_mask[j];
            else 
                lossWeight = 1 - attnWeight;
            return `<span id="${j}" class="transition-all ${className}" style="background-color: ${getGreenOpacity(lossWeight, attnWeight, className)}">${escapedStr}</span>`;
        }).join(' ');
    };        
    document.getElementById('input_strings_display').innerHTML = createSpanWithBG(globalHDF5Data.input_strings, 'input_strings');
    document.getElementById('input_ids_display').innerHTML = createSpan(globalHDF5Data.input_ids, 'input_ids', ', ');
    document.getElementById('label_ids_display').innerHTML = createSpan(globalHDF5Data.labels, 'label_ids', ', ');
    
    // Special handling for strings as they are not arrays of numbers
    document.getElementById('label_strings_display').innerHTML = createSpanWithBG(globalHDF5Data.label_strings, 'label_strings');
    registerHoverEvents();
}
        
document.getElementById('stringLengthSelector').addEventListener('change', function() {
    loadData(this.value-1);
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
