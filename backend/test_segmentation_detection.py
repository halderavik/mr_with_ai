#!/usr/bin/env python3
"""
Test script to verify segmentation detection works correctly.
"""

def test_segmentation_detection():
    """Test the segmentation detection logic."""
    
    print("=== Testing Segmentation Detection ===\n")
    
    # Test cases
    test_cases = [
        "can you run van westendrop analysis by age group",
        "run van westendorp by age",
        "analyze by age groups",
        "van westendorp analysis by gender",
        "run analysis by income level",
        "just run van westendorp analysis",  # No segmentation
        "van westendorp by age group please"
    ]
    
    for question in test_cases:
        print(f"Testing: '{question}'")
        
        # Simulate the detection logic
        question_lower = question.lower()
        segmentation_requested = None
        
        # Look for "by" patterns in the question
        if "by" in question_lower:
            parts = question_lower.split("by")
            if len(parts) > 1:
                potential_seg = parts[1].strip()
                # Clean up common words
                potential_seg = potential_seg.replace("group", "").replace("groups", "").replace("s", "").strip()
                segmentation_requested = potential_seg
                print(f"  ✓ Detected segmentation: '{potential_seg}'")
        else:
            print(f"  ✗ No segmentation detected")
        
        print()
    
    print("Segmentation detection test completed!")

if __name__ == "__main__":
    test_segmentation_detection() 