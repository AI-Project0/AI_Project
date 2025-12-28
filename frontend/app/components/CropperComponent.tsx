"use client";

import React, { useState, useCallback } from "react";
import Cropper, { Area } from "react-easy-crop";

interface CropperProps {
    imageSrc: string;
    aspectRatio: number; // e.g., 35/45
    onCropComplete: (croppedAreaPixels: Area) => void;
}

const CropperComponent: React.FC<CropperProps> = ({
    imageSrc,
    aspectRatio,
    onCropComplete,
}) => {
    const [crop, setCrop] = useState({ x: 0, y: 0 });
    const [zoom, setZoom] = useState(1);

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
                    containerStyle: { background: "#1a1a1a" },
                }}
            />
        </div>
    );
};

export default CropperComponent;
