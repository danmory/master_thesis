// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FeesTemplate
 * @dev Represents the additional fees details from Section 9 of the Short-Term Rental Agreement.
 */
contract FeesTemplate {
    bool public cleaningFeeRequired;
    uint256 public cleaningFeeAmount;

    bool public taxesRequired; // Assuming taxes might be a separate line item
    uint256 public taxesAmount;

    // For 'Other' fees
    bool public otherFeeRequired;
    string public otherFeeDescription;
    uint256 public otherFeeAmount;

    // --- Fee Details ---
    // cleaningFeeRequired: True if a cleaning fee is charged.
    // cleaningFeeAmount: The amount of the cleaning fee.
    // taxesRequired: True if specific taxes are listed as a fee.
    // taxesAmount: The amount of the taxes fee.
    // otherFeeRequired: True if any 'Other' fees are specified.
    // otherFeeDescription: Description of the 'Other' fee.
    // otherFeeAmount: Amount of the 'Other' fee.

    // Note: Actual implementation would involve functions for collecting these fees at execution.
    // This template primarily defines the data structure.
}