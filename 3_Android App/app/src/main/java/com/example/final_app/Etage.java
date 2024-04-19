package com.example.final_app;

import android.os.Parcelable;
import android.os.Parcel;

public class Etage implements Parcelable {
    int numero, victimes;
    int[] cameras;

    public Etage(int numero) {
        this.numero = numero;
        this.victimes = 2;
        this.cameras = new int[] {1, 2, 3};
    }

    //////////////////
    // Parcelable

    // Constructor for Parcelable
    protected Etage(Parcel in) {
        numero = in.readInt();
        victimes = in.readInt();
        cameras = in.createIntArray();
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeInt(numero);
        dest.writeInt(victimes);
        dest.writeIntArray(cameras);
    }

    @Override
    public int describeContents() {
        return 0;
    }

    // Creator constant for Parcelable
    public static final Creator<Etage> CREATOR = new Creator<Etage>() {
        @Override
        public Etage createFromParcel(Parcel in) {
            return new Etage(in);
        }

        @Override
        public Etage[] newArray(int size) {
            return new Etage[size];
        }
    };
}
