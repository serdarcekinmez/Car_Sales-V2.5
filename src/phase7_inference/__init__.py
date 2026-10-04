"""
Phase 7 inference layer: one vehicle -> Model A call list + Model B price, no training.

Entry points (imported from their modules; this package imports nothing at load, so
vehicle_input stays importable without the model dependencies):
  phase7_inference.bundle   BundleConfig, load_bundle, cached_bundle
  phase7_inference.predict  predict_vehicle, replay_history, response_json, InputError
  phase7_inference.filters  parse_filters, describe_filters, FILTER_VERSION
"""
