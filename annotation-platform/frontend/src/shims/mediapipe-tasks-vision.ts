// The annotation workbench does not use drei's optional FaceLandmarker feature.
// This shim prevents Vite from resolving the optional MediaPipe package during
// dependency scanning on Windows. If FaceLandmarker is enabled in the future,
// remove this alias and install a Vite-compatible MediaPipe package.
export const FilesetResolver = {};
export const FaceLandmarker = {};
export default {};
