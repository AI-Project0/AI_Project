"use client";

import React, { useState, useCallback } from "react";
import Cropper, { Area } from "react-easy-crop";

interface CropperProps {
    imageSrc: string;
    aspectRatio: number; // e.g., 35/45
    onCropComplete: (croppedAreaPixels: Area) => void;
    initialCrop?: { x: number; y: number };
    initialZoom?: number;
}

const CropperComponent: React.FC<CropperProps> = ({
    imageSrc,
    aspectRatio,
    onCropComplete,
    initialCrop = { x: 0, y: 0 },
    initialZoom = 1,
}) => {
    const [crop, setCrop] = useState(initialCrop);
    const [zoom, setZoom] = useState(initialZoom);

    // Sync with initialCrop if it changes (e.g., after API response)
    React.useEffect(() => {
        if (initialCrop) setCrop(initialCrop);
    }, [initialCrop]);

    React.useEffect(() => {
        if (initialZoom) setZoom(initialZoom);
    }, [initialZoom]);

    const onCropChange = (crop: { x: number; y: number }) => {
        setCrop(crop);
    };

    const onZoomChange = (zoom: number) => {
        setZoom(zoom);
    };

    const handleCropComplete = useCallback(
        (_croppedArea: Area, croppedAreaPixels: Area) => {
            onCropComplete(croppedAreaPixels);
        },
        [onCropComplete]
    );

    return (
        <div className="relative w-full h-full bg-gray-900 rounded-lg overflow-hidden shadow-inner">
            <Cropper
                image={imageSrc}
                crop={crop}
                zoom={zoom}
                aspect={aspectRatio}
                onCropChange={onCropChange}
                onCropComplete={handleCropComplete}
                onZoomChange={onZoomChange}
                style={{
                    containerStyle: { background: "#1a1a1a", width: "100%", height: "100%" },
                    mediaStyle: { width: 'auto', height: '100%' }
                }}
            />
        </div>
    );
};

export default CropperComponent;
