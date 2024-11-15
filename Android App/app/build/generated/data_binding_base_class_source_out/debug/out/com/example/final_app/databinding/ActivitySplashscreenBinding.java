// Generated by view binder compiler. Do not edit!
package com.example.final_app.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.constraintlayout.widget.ConstraintLayout;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.example.final_app.R;
import java.lang.NullPointerException;
import java.lang.Override;
import java.lang.String;

public final class ActivitySplashscreenBinding implements ViewBinding {
  @NonNull
  private final ConstraintLayout rootView;

  @NonNull
  public final TextView nomDesEncadrants;

  @NonNull
  public final TextView nomDuProjet;

  @NonNull
  public final TextView versionApp;

  private ActivitySplashscreenBinding(@NonNull ConstraintLayout rootView,
      @NonNull TextView nomDesEncadrants, @NonNull TextView nomDuProjet,
      @NonNull TextView versionApp) {
    this.rootView = rootView;
    this.nomDesEncadrants = nomDesEncadrants;
    this.nomDuProjet = nomDuProjet;
    this.versionApp = versionApp;
  }

  @Override
  @NonNull
  public ConstraintLayout getRoot() {
    return rootView;
  }

  @NonNull
  public static ActivitySplashscreenBinding inflate(@NonNull LayoutInflater inflater) {
    return inflate(inflater, null, false);
  }

  @NonNull
  public static ActivitySplashscreenBinding inflate(@NonNull LayoutInflater inflater,
      @Nullable ViewGroup parent, boolean attachToParent) {
    View root = inflater.inflate(R.layout.activity_splashscreen, parent, false);
    if (attachToParent) {
      parent.addView(root);
    }
    return bind(root);
  }

  @NonNull
  public static ActivitySplashscreenBinding bind(@NonNull View rootView) {
    // The body of this method is generated in a way you would not otherwise write.
    // This is done to optimize the compiled bytecode for size and performance.
    int id;
    missingId: {
      id = R.id.nom_des_encadrants;
      TextView nomDesEncadrants = ViewBindings.findChildViewById(rootView, id);
      if (nomDesEncadrants == null) {
        break missingId;
      }

      id = R.id.nom_du_projet;
      TextView nomDuProjet = ViewBindings.findChildViewById(rootView, id);
      if (nomDuProjet == null) {
        break missingId;
      }

      id = R.id.version_app;
      TextView versionApp = ViewBindings.findChildViewById(rootView, id);
      if (versionApp == null) {
        break missingId;
      }

      return new ActivitySplashscreenBinding((ConstraintLayout) rootView, nomDesEncadrants,
          nomDuProjet, versionApp);
    }
    String missingId = rootView.getResources().getResourceName(id);
    throw new NullPointerException("Missing required view with ID: ".concat(missingId));
  }
}
