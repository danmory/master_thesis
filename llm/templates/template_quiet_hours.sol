// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title QuietHoursTemplate
 * @dev Represents the quiet hours policy details from Section 14 of the Short-Term Rental Agreement.
 */
contract QuietHoursTemplate {
    bool public quietHoursRequired;
    string public quietHoursStartTime; // e.g., "22:00"
    // End time is specified as sunrise in the agreement.

    // --- Quiet Hours Policy Details ---
    // quietHoursRequired: True if specific quiet hours are enforced, false otherwise.
    // quietHoursStartTime: The time quiet hours begin each night.

    // Note: The agreement mentions quiet hours continue until sunrise, which is variable.
    // This template primarily defines the data structure for the start time.
}