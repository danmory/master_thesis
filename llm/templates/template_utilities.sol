// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title UtilitiesTemplate
 * @dev Represents the utility responsibilities from Section 6 of the Short-Term Rental Agreement.
 */
contract UtilitiesTemplate {
    // Describes which utilities the Tenant is responsible for.
    // The Landlord is responsible for all others by default according to the agreement.
    string public tenantResponsibleUtilitiesDescription;

    // Note: Actual implementation might involve more structured data, e.g., boolean flags for common utilities.
    // This template primarily defines the data structure based on the text.
}