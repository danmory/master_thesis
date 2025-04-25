// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title SublettingTemplate
 * @dev Represents the subletting policy details from Section 12 of the Short-Term Rental Agreement.
 */
contract SublettingTemplate {
    bool public sublettingAllowed;
    bool public landlordApprovalRequired; // True if landlord approval is needed for subtenants.

    // --- Subletting Policy Details ---
    // sublettingAllowed: True if the tenant can sublet, false otherwise.
    // landlordApprovalRequired: Applicable only if sublettingAllowed is true. Indicates if landlord must approve subtenants.

    // Note: This template primarily defines the data structure.
}