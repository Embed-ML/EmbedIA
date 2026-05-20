/*
 * Test file for the refactored dfixed implementation
 * This file demonstrates the new 32-bit dfixed with extended range
 */

#include <stdio.h>
#include "fixed.h"

int main() {
    printf("=== DFIXED 32-bit Extended Range Test ===\n\n");
    
    // Test basic constants
    printf("Fixed constants:\n");
    printf("FIX_ONE = %d (%.6f)\n", FIX_ONE, FX2FL(FIX_ONE));
    printf("FIX_FRC_SZ = %d bits\n", FIX_FRC_SZ);
    printf("FIX_INT_SZ = %d bits\n", FIX_INT_SZ);
    
    printf("\nDFixed constants:\n");
    printf("DFIX_ONE = %d (%.6f)\n", DFIX_ONE, DFX2FL(DFIX_ONE));
    printf("FIX_DFRC_SZ = %d bits\n", FIX_DFRC_SZ);
    printf("FIX_DINT_SZ = %d bits\n", FIX_DINT_SZ);
    
    // Test conversions
    printf("\n=== Conversion Tests ===\n");
    fixed fx_val = FL2FX(3.14159f);
    dfixed dfx_val = FL2DFX(3.14159f);
    
    printf("Original float: 3.14159\n");
    printf("Fixed (17 frac bits): %d -> %.6f\n", fx_val, FX2FL(fx_val));
    printf("DFixed (13 frac bits): %d -> %.6f\n", dfx_val, DFX2FL(dfx_val));
    
    // Test fixed to dfixed conversion
    dfixed converted = FIXED_TO_DFIXED(fx_val);
    printf("Fixed->DFixed: %d -> %.6f\n", converted, DFX2FL(converted));
    
    // Test dfixed to fixed conversion
    fixed back_converted = DFIXED_TO_FIXED(dfx_val);
    printf("DFixed->Fixed: %d -> %.6f\n", back_converted, FX2FL(back_converted));
    
    // Test arithmetic operations
    printf("\n=== Arithmetic Tests ===\n");
    dfixed a = FL2DFX(2.5f);
    dfixed b = FL2DFX(1.5f);
    
    printf("a = %.6f, b = %.6f\n", DFX2FL(a), DFX2FL(b));
    printf("a + b = %.6f\n", DFX2FL(DFIXED_ADD(a, b)));
    printf("a - b = %.6f\n", DFX2FL(DFIXED_SUB(a, b)));
    printf("a * b = %.6f\n", DFX2FL(DFIXED_MUL(a, b)));
    printf("a / b = %.6f\n", DFX2FL(DFIXED_DDIV(a, b)));
    
    // Test range comparison
    printf("\n=== Range Comparison ===\n");
    printf("Fixed max value: %.6f\n", FX2FL(FIX_MAX));
    printf("DFixed max value: %.6f\n", DFX2FL(DFIX_MAX));
    printf("DFixed has %d times more integer range\n", 1 << 4);
    
    // Test precision comparison
    printf("\n=== Precision Comparison ===\n");
    float test_val = 0.123456789f;
    fixed fx_test = FL2FX(test_val);
    dfixed dfx_test = FL2DFX(test_val);
    
    printf("Original: %.9f\n", test_val);
    printf("Fixed:    %.9f (error: %.9f)\n", FX2FL(fx_test), test_val - FX2FL(fx_test));
    printf("DFixed:   %.9f (error: %.9f)\n", DFX2FL(dfx_test), test_val - DFX2FL(dfx_test));
    
    printf("\n=== Test Complete ===\n");
    return 0;
}