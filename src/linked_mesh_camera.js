const plot = document.getElementById("{plot_id}");
const storageKey = plot.id + "-canonical-camera";
const sign = {linked_yz_sign};
let syncing = false;

// A 180-degree turn about X maps canonical +Y-up and posed -Y-up views.
function otherCamera(camera) {
    const converted = {};
    for (const key of ["eye", "up", "center"]) {
        if (camera[key]) {
            const { x, y, z } = camera[key];
            converted[key] = { x, y: sign * y, z: sign * z };
        }
    }
    if (camera.projection) {
        converted.projection = { type: camera.projection.type };
    }
    return converted;
}

function updateCameras(changes) {
    syncing = true;
    Promise.resolve(Plotly.relayout(plot, changes)).finally(() => { syncing = false; });
}

function syncFrom(scene, current, saveSource = false) {
    const target = scene === "scene" ? "scene2" : "scene";
    const peer = otherCamera(current);
    try {
        sessionStorage.setItem(storageKey, JSON.stringify(scene === "scene" ? current : peer));
    } catch {
        // Some browser modes disable storage. Linked interaction still works.
    }
    const changes = { [target + ".camera"]: peer };
    if (saveSource) changes[scene + ".camera"] = current;
    updateCameras(changes);
}

plot.on("plotly_relayout", (changes) => {
    if (syncing) return;
    const keys = Object.keys(changes);
    const scene = ["scene", "scene2"].find((name) =>
        keys.some((key) => key === name + ".camera" || key.startsWith(name + ".camera."))
    );
    if (!scene) return;
    const current = changes[scene + ".camera"] ?? plot.layout[scene].camera;
    if (current) syncFrom(scene, current);
});

// Plotly can emit its relayout event before the perspective wheel zoom has updated the camera.
let wheelTimer;
plot.addEventListener("wheel", (event) => {
    const scene = ["scene", "scene2"].find((name) =>
        plot._fullLayout[name]?._scene?.container.contains(event.target)
    );
    if (!scene) return;
    clearTimeout(wheelTimer);
    wheelTimer = setTimeout(() => {
        syncFrom(scene, plot._fullLayout[scene]._scene.getCamera(), true);
    }, 100);
}, { capture: true, passive: true });

try {
    const saved = JSON.parse(sessionStorage.getItem(storageKey));
    if (saved?.eye) {
        updateCameras({ "scene.camera": saved, "scene2.camera": otherCamera(saved) });
    }
} catch {
    // Start with the layout cameras when storage is unavailable.
}
