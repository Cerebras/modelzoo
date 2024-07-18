function getGreenOpacity(lossWeight, attnWeight, className) {
    if(lossWeight > 0)
        return `rgba(0, 255, 0, ${lossWeight/1.5})`;
    else if(!attnWeight && !(lossWeight > 0))
        return 'rgba(255, 0, 0, 0.8)'
}

function addMouseEvents(element, inputStringsSpans, labelStringsSpans) {
    element.addEventListener('mouseover', function() {
        inputStringsSpans.forEach(s => {
            if (s.id !== element.id) {
                s.style.backgroundColor = "#ececec";
            }
        });
        labelStringsSpans.forEach(s => {
            if (s.id !== element.id) {
                s.style.backgroundColor = "#ececec";
            }
        });
        
        const correspondingInputId = document.querySelector(`#input_ids_display span[id="${element.id}"]`);
        const correspondingInputStr = document.querySelector(`#input_strings_display span[id="${element.id}"]`);
        const correspondingOutputId = document.querySelector(`#label_ids_display span[id="${element.id}"]`);
        const correspondingOutputStr = document.querySelector(`#label_strings_display span[id="${element.id}"]`);
        const attnWeight = globalHDF5Data.attention_mask[element.id];
        var lossWeight;
        if('loss_mask' in globalHDF5Data)
            lossWeight = globalHDF5Data.loss_mask[element.id];
        else 
            lossWeight = 1 - attnWeight;
        var colorToApply;
        if(!lossWeight && attnWeight)
            colorToApply = "#999";
        else
            colorToApply = getGreenOpacity(lossWeight, attnWeight, '');
        element.style.backgroundColor = colorToApply
        correspondingInputId.style.backgroundColor = colorToApply;
        correspondingOutputId.style.backgroundColor = colorToApply;
        correspondingOutputStr.style.backgroundColor = colorToApply;
        correspondingInputStr.style.backgroundColor = colorToApply;
        scrollIntoViewIfNeeded(correspondingInputId, document.getElementById('input_ids_display'));
        scrollIntoViewIfNeeded(correspondingOutputId, document.getElementById('label_ids_display'));
        var popup = document.querySelector('.popup');
        if (popup) {
            popup.remove();
        }
        popup = document.createElement('div');
        const attention_mask = globalHDF5Data.attention_mask[element.id];
        var loss_mask;
        if('loss_mask' in globalHDF5Data)
            loss_mask = globalHDF5Data.loss_mask[element.id];
        else 
        loss_mask = 1 - attnWeight;
        var position_id = element.id;
        if ('position_ids' in globalHDF5Data) {
            position_id = globalHDF5Data.position_ids[element.id]
        }
        if(globalHDF5Data.images_bitmap[element.id])
            popup.innerHTML += 
                `<img class="popup-image" src="/images/${globalHDF5Data.image_paths[globalHDF5Data.images_bitmap[element.id]-1]}" alt="Popup Image">`
        popup.className = 'popup';
        popup.innerHTML += `
                <div class="popup-text">
                    <p><b>Index in MSL</b>: ${element.id}</p>
                    <p><b>Loss Weight</b>: ${loss_mask}</p>
                    <p><b>Attention Mask</b>: ${attention_mask}
                    <p><b>Position ID</b>: ${position_id}
                    <p><b>Drop Mask</b>: 0</p>
                </div>
            </div>
        `;
        document.body.appendChild(popup);

        const rect = element.getBoundingClientRect();
        const containerRect = document.getElementsByClassName('container')[0].getBoundingClientRect();
        popup.style.left = rect.x + 350 > containerRect.width ? 
            `${rect.left + window.scrollX - 250}px` : 
            `${rect.left + window.scrollX + 100}px`;
        popup.style.top = `${rect.top + window.scrollY - 10}px`;
        const popupEl = document.getElementsByClassName('popup-text')[0];
        if(!globalHDF5Data.images_bitmap[element.id])
                popupEl.style.paddingLeft = "16px";
    });

    element.addEventListener('mouseout', function() {
        inputStringsSpans.forEach(s => {
            s.style.backgroundColor = s.dataset.originalColor;
        });
        labelStringsSpans.forEach(s => {
            s.style.backgroundColor = s.dataset.originalColor;
        });
        // Reset the class of the corresponding span in input_ids_display
        const correspondingInputId = document.querySelector(`#input_ids_display span[id="${element.id}"]`);
        const correspondingOutputId = document.querySelector(`#label_ids_display span[id="${element.id}"]`)
        correspondingInputId.style.backgroundColor = "#ececec"; // Restore original class
        correspondingOutputId.style.backgroundColor = "#ececec"; // Restore original class
        const popup = document.querySelector('.popup');
        if (popup) {
            popup.remove();
        }
    });

}

function registerHoverEvents() {
    const inputStringsSpans = document.querySelectorAll('#input_strings_display span');
    const labelStringsSpans = document.querySelectorAll('#label_strings_display span');
    
    inputStringsSpans.forEach(span => {
        span.dataset.originalColor = span.style.backgroundColor;
    });
    labelStringsSpans.forEach(span => {
        span.dataset.originalColor = span.style.backgroundColor;
    });
    
    inputStringsSpans.forEach(el => addMouseEvents(el, inputStringsSpans, labelStringsSpans));
    labelStringsSpans.forEach(el => addMouseEvents(el, inputStringsSpans, labelStringsSpans));
    document.querySelectorAll("#input_ids_display span").forEach(el => addMouseEvents(el, inputStringsSpans, labelStringsSpans));
    document.querySelectorAll("#label_ids_display span").forEach(el => addMouseEvents(el, inputStringsSpans, labelStringsSpans));
    document.querySelectorAll("#label_strings_display span").forEach(el => addMouseEvents(el, inputStringsSpans, labelStringsSpans));
};
