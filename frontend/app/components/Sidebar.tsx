"use client";

import React from "react";
import {
    Check,
    User,
    Briefcase,
    Shirt,
    Stamp,
    Wand2
} from "lucide-react";

export type SizeId = "1inch" | "2inch_head" | "2inch_half";
export type BgColor = "#FFFFFF" | "#4B89DC" | "#808080";
export type OutfitType = "original" | "suit_male" | "suit_female";

interface SidebarProps {
    settings: {
        sizeId: SizeId;
        bgColor: BgColor;
        outfitType: OutfitType;
    };
    setSettings: React.Dispatch<React.SetStateAction<{
        sizeId: SizeId;
        bgColor: BgColor;
        outfitType: OutfitType;
    }>>;
    onGenerate: () => void;
    isGenerating: boolean;
    hasImage: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({
    settings,
    setSettings,
    onGenerate,
    isGenerating,
    hasImage
}) => {

    const updateSetting = (key: string, value: any) => {
        setSettings(prev => ({ ...prev, [key]: value }));
    };

    return (
        <div className="h-full w-full bg-cream-card p-6 md:p-8 shadow-soft-xl flex flex-col gap-6 md:gap-8 lg:rounded-l-[40px] border-l border-cream-border/50 overflow-y-auto">

            {/* Header - Clean & Minimalist */}
            <div className="flex items-center gap-3 mb-2 shrink-0">
                <div className="bg-cream-accent p-3 rounded-2xl shadow-sm rotate-3">
                    <Wand2 className="text-cream-text w-6 h-6" />
                </div>
                <div>
                    <h1 className="text-xl md:text-2xl font-bold text-cream-text tracking-tight">AI 智慧證件照</h1>
                    <p className="text-[10px] md:text-xs text-cream-text-light font-medium tracking-widest uppercase">Pro ID Generator</p>
                </div>
            </div>

            {/* Anti-Fumble Container */}
            <div className={`flex flex-col gap-6 md:gap-8 transition-all duration-500 shrink-0 ${isGenerating ? "opacity-50 pointer-events-none grayscale cursor-not-allowed" : ""}`}>

                {/* Section 1: Size */}
                <div className="space-y-4">
                    <h3 className="text-xs md:text-sm font-bold text-cream-text-light uppercase tracking-wider flex items-center gap-2 ml-1">
                        <Stamp className="w-4 h-4" /> 尺寸選擇 (Size)
                    </h3>
                    <div className="grid grid-cols-1 gap-3">
                        {[
                            { id: "1inch", label: "1 吋", desc: "28x35mm", usage: "1 吋 (適用：駕照、身分證、體檢)" },
                            { id: "2inch_head", label: "2 吋大頭", desc: "35x45mm", usage: "2 吋 (適用：護照、台胞證)" },
                            { id: "2inch_half", label: "2 吋半身", desc: "42x47mm", usage: "專業求職、履歷面試" },
                        ].map((opt) => (
                            <button
                                key={opt.id}
                                onClick={() => updateSetting("sizeId", opt.id)}
                                className={`flex items-center gap-4 p-3 md:p-4 rounded-2xl transition-all duration-300 border-2 text-left ${settings.sizeId === opt.id
                                    ? "bg-cream-accent border-cream-accent text-cream-text font-bold shadow-soft-md scale-[1.02]"
                                    : "bg-cream-bg border-cream-border text-cream-text hover:border-cream-accent hover:bg-white"
                                    }`}
                            >
                                <div className="flex-1">
                                    <div className="flex justify-between items-center mb-1">
                                        <span className="text-sm md:text-base">{opt.label}</span>
                                        <span className="text-[10px] opacity-60">{opt.desc}</span>
                                    </div>
                                    <p className="text-[10px] md:text-xs font-medium opacity-70 leading-tight">{opt.usage}</p>
                                </div>
                                {settings.sizeId === opt.id && <Check className="w-5 h-5 text-cream-text" />}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Section 2: Background */}
                <div className="space-y-4">
                    <h3 className="text-xs md:text-sm font-bold text-cream-text-light uppercase tracking-wider flex items-center gap-2 ml-1">
                        <div className="w-4 h-4 rounded-full border border-cream-text-light bg-white"></div> 背景顏色 (Background)
                    </h3>
                    <div className="flex flex-col gap-4">
                        <div className="flex gap-4 px-2">
                            {[
                                { color: "#FFFFFF", name: "White", ring: "ring-cream-border", usage: "白色 (標準：身分證、護照)" },
                                { color: "#4B89DC", name: "Blue", ring: "ring-blue-200", usage: "藍色 (求職、識別證)" },
                                { color: "#808080", name: "Gray", ring: "ring-gray-300", usage: "灰色 (專業形象、形象照)" },
                            ].map((opt) => (
                                <button
                                    key={opt.color}
                                    onClick={() => updateSetting("bgColor", opt.color)}
                                    className={`w-12 h-12 md:w-14 md:h-14 rounded-full shadow-sm flex items-center justify-center transition-all duration-300 hover:scale-110 ${opt.ring
                                        } relative border-4 border-white ${settings.bgColor === opt.color ? 'scale-110 shadow-md' : 'opacity-80'}`}
                                    style={{ backgroundColor: opt.color }}
                                >
                                    {settings.bgColor === opt.color && (
                                        <div className="bg-white/20 p-1 rounded-full backdrop-blur-sm">
                                            <Check className={`w-5 h-5 ${opt.color === "#FFFFFF" ? "text-gray-800" : "text-white"}`} />
                                        </div>
                                    )}
                                </button>
                            ))}
                        </div>
                        <div className="px-2">
                            <p className="text-[10px] md:text-[11px] font-bold text-cream-text-light/80 italic">
                                {settings.bgColor === "#FFFFFF" && "✓ 白色 (標準：身分證、護照)"}
                                {settings.bgColor === "#4B89DC" && "✓ 藍色 (求職、識別證)"}
                                {settings.bgColor === "#808080" && "✓ 灰色 (專業形象、形象照)"}
                            </p>
                        </div>
                    </div>
                </div>

                {/* Section 3: Outfit */}
                <div className="space-y-4">
                    <h3 className="text-xs md:text-sm font-bold text-cream-text-light uppercase tracking-wider flex items-center gap-2 ml-1">
                        <Shirt className="w-4 h-4" /> 智慧服裝 (Outfit)
                    </h3>
                    <div className="grid grid-cols-1 gap-2">
                        {[
                            { id: "original", label: "保留原衣", icon: <User className="w-4 h-4" /> },
                            { id: "suit_male", label: "男士西裝", icon: <Briefcase className="w-4 h-4" /> },
                            { id: "suit_female", label: "女士套裝", icon: <Briefcase className="w-4 h-4" /> },
                        ].map((opt) => (
                            <button
                                key={opt.id}
                                onClick={() => updateSetting("outfitType", opt.id)}
                                className={`flex items-center gap-3 p-3 rounded-xl border-2 transition-all duration-300 ${settings.outfitType === opt.id
                                    ? "border-cream-accent bg-cream-accent/30 text-cream-text shadow-sm"
                                    : "border-cream-border bg-white text-cream-text-light hover:border-cream-accent"
                                    }`}
                            >
                                <div className={`p-2 rounded-lg ${settings.outfitType === opt.id ? "bg-cream-accent" : "bg-cream-bg"}`}>
                                    {opt.icon}
                                </div>
                                <span className={`font-bold text-xs md:text-sm ${settings.outfitType === opt.id ? "text-cream-text" : "text-cream-text/80"}`}>{opt.label}</span>
                                {settings.outfitType === opt.id && <Check className="ml-auto w-4 h-4 text-cream-text" />}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Generate Button */}
            <div className="mt-auto pt-4 shrink-0">
                <button
                    onClick={onGenerate}
                    disabled={!hasImage || isGenerating}
                    className={`w-full h-16 md:h-20 rounded-full flex items-center justify-center gap-3 font-bold text-lg md:text-xl shadow-lg transition-all duration-300 transform ${!hasImage || isGenerating
                        ? "bg-cream-border text-cream-text-light cursor-not-allowed opacity-50"
                        : "bg-gradient-to-r from-cream-accent to-cream-accent-hover text-cream-text hover:shadow-orange-200/50 hover:-translate-y-1 active:scale-95 active:translate-y-0"
                        }`}
                >
                    {isGenerating ? (
                        <>
                            <div className="animate-spin rounded-full h-5 w-5 md:h-6 md:w-6 border-b-2 border-cream-text"></div>
                            <span className="tracking-widest text-[10px] md:text-xs opacity-60">處理中...</span>
                        </>
                    ) : (
                        <>
                            <Wand2 className="w-5 h-5 md:w-6 md:w-6" />
                            <span className="tracking-wide">開始生成</span>
                        </>
                    )}
                </button>
            </div>
        </div>
    );
};

export default Sidebar;
