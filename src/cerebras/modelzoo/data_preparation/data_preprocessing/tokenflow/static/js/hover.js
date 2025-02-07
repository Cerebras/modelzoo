function getGreenOpacity(lossWeight, attnWeight, className) {
    if (lossWeight > 0)
        return `rgba(0, 255, 0, ${lossWeight / 1.5})`;
    else if (!attnWeight && !(lossWeight > 0))
        return 'rgba(255, 0, 0, 0.8)'
}

function createHoverPopup(add_image, image_path, element_id, loss_mask, attention_mask, position_id) {
    const popup = document.createElement('div');

    if(add_image){
        popup.innerHTML +=
            `<img class="popup-image" src="/images/${image_path}" alt="Popup Image">`
    }
    popup.className = 'popup';
    popup.innerHTML += `
    <div class="popup-text">
        <p><b>Index in MSL</b>: ${element_id}</p>
        <p><b>Loss Weight</b>: ${loss_mask}</p>
        <p><b>Attention Mask</b>: ${attention_mask}
        <p><b>Position ID</b>: ${position_id}
        <p><b>Drop Mask</b>: 0</p>
    </div>
</div>
`;
    return popup;
}

function getWeightsAndMasks(mode, elementId) {
    let attnWeight, lossWeight, loss_mask, position_id;

    if (mode === 'C') {
        attnWeight = globalHDF5Data.chosen_attention_mask[elementId];
        lossWeight = 'chosen_loss_mask' in globalHDF5Data ? globalHDF5Data.chosen_loss_mask[elementId] : 1 - attnWeight;
        loss_mask = 'chosen_loss_mask' in globalHDF5Data ? globalHDF5Data.chosen_loss_mask[elementId] : 1 - attnWeight;
        position_id = 'chosen_position_ids' in globalHDF5Data ? globalHDF5Data.chosen_position_ids[elementId] : elementId;
    } else if (mode === 'R') {
        attnWeight = globalHDF5Data.rejected_attention_mask[elementId];
        lossWeight = 'rejected_loss_mask' in globalHDF5Data ? globalHDF5Data.rejected_loss_mask[elementId] : 1 - attnWeight;
        loss_mask = 'rejected_loss_mask' in globalHDF5Data ? globalHDF5Data.rejected_loss_mask[elementId] : 1 - attnWeight;
        position_id = 'rejected_position_ids' in globalHDF5Data ? globalHDF5Data.rejected_position_ids[elementId] : elementId;
    } else {
        attnWeight = globalHDF5Data.attention_mask[elementId];
        lossWeight = 'loss_mask' in globalHDF5Data ? globalHDF5Data.loss_mask[elementId] : 1 - attnWeight;
        loss_mask = 'loss_mask' in globalHDF5Data ? globalHDF5Data.loss_mask[elementId] : 1 - attnWeight;
        position_id = 'position_ids' in globalHDF5Data ? globalHDF5Data.position_ids[elementId] : elementId;
    }

    return { attnWeight, lossWeight, loss_mask, position_id };
}

function addMouseEvents(element, inputStringsSpans, labelStringsSpans, section) {
    element.addEventListener('mouseover', function () {
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

        let popup = document.querySelector('.popup');
        if (popup) {
            popup.remove();
        }

        var lossWeight, loss_mask, attention_mask, attnWeight, colorToApply;
        var { attnWeight, lossWeight, loss_mask, position_id } = getWeightsAndMasks(globalDPOMode, element.id);

        if(globalDPOMode == 'C'){
            attention_mask = globalHDF5Data.chosen_attention_mask[element.id];
        } else if(globalDPOMode == 'R'){
            attention_mask = globalHDF5Data.rejected_attention_mask[element.id];
        } else{
            attention_mask = globalHDF5Data.attention_mask[element.id];
        }

        if (!lossWeight && attnWeight)
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

        if (section == 'input') {
            const image_path = globalHDF5Data.image_paths[globalHDF5Data.images_bitmap[element.id] - 1]
            const add_image = globalHDF5Data.images_bitmap[element.id]
            popup = createHoverPopup(add_image, image_path, element.id, loss_mask, attention_mask, position_id)
            if (popup instanceof Node){
                document.body.appendChild(popup);
            } else{
                console.log('Failed to create popup - not a valid DOM node.')
            }
        }

        if (section == 'label') {
            const bitmapLength = globalHDF5Data.images_bitmap.length
            const tempBitMask = [...globalHDF5Data.images_bitmap]

            for (let i = 0; i < bitmapLength - 1; i++) {
                tempBitMask[i] = tempBitMask[i + 1]
            }

            const image_path = globalHDF5Data.image_paths[tempBitMask[element.id] - 1]
            const add_image = tempBitMask[element.id]
            popup = createHoverPopup(add_image, image_path, element.id, loss_mask, attention_mask, position_id)
            if (popup instanceof Node){
                document.body.appendChild(popup);
            } else{
                console.log('Failed to create popup - not a valid DOM node.')
            }
        }

        const rect = element.getBoundingClientRect();
        const containerRect = document.getElementsByClassName('container')[0].getBoundingClientRect();
        popup.style.left = rect.x + 350 > containerRect.width ?
            `${rect.left + window.scrollX - 250}px` :
            `${rect.left + window.scrollX + 100}px`;
        popup.style.top = `${rect.top + window.scrollY - 10}px`;
        const popupEl = document.getElementsByClassName('popup-text')[0];
        if (!globalHDF5Data.images_bitmap[element.id])
            popupEl.style.paddingLeft = "16px";
    });

    element.addEventListener('mouseout', function () {
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

    inputStringsSpans.forEach(el => addMouseEvents(el, inputStringsSpans, labelStringsSpans, 'input'));
    labelStringsSpans.forEach(el => addMouseEvents(el, inputStringsSpans, labelStringsSpans, 'label'));
    document.querySelectorAll("#input_ids_display span").forEach(el => addMouseEvents(el, inputStringsSpans, labelStringsSpans, 'input'));
    document.querySelectorAll("#label_ids_display span").forEach(el => addMouseEvents(el, inputStringsSpans, labelStringsSpans, 'label'));
};
